"""Full-pretraining-corpus PCA of centered log1p counts, then frozen Hao projection.

Run this file directly. Edit the constants below; there are no command-line flags.
For a tiny smoke test set N_GENES=128, N_COMPONENTS=16,
LIMIT_N_TRAIN_CELLS=256 and LIMIT_N_HAO_CELLS=64. Restore defaults for the full fit.
Smoke configurations have separate artifact directories and never publish to the
full Hao representation path. Merely limiting cells still allocates a covariance
of size N_GENES by N_GENES, so reduce N_GENES too for a lightweight smoke test.
Re-running resumes scans and skips completed stages. Ctrl+C/SIGTERM finishes the
current batch and checkpoints; SIGKILL resumes from the last periodic checkpoint.
An interrupted eigensolve restarts that stage, without repeating either data scan.
Keep roughly 60 GB of disk space free for atomic covariance checkpoints. A new
checkpoint is fully saved before the previous one is removed. Temporary arrays
are gene-block sized; the full float64 covariance is the main RAM allocation.

Cross-products use dense PyTorch matrix multiplications on the configured device;
means, float64 accumulation, centering, eigensolve and Hao projection use CPU.
Float64 GPU multiplication preserves the original precision. Optional float32
products trade precision for speed, with TF32 disabled and float64 accumulation.
FROM_SCRATCH deletes all this script's fit directories and published Hao output;
set it back to False afterwards to resume subsequent runs.
No count normalization, gene standardization, or PCA whitening.
This script only generates representations; it does not modify the Hao collector.
"""

from pathlib import Path

# ----------------------------- Configuration -----------------------------
TRAIN_SHARDS = Path('/workspace/data/shuffled_shards')
REFERENCE_VAR = Path('/workspace/data/var_sorted.csv')
HAO_H5AD = Path('/workspace/data/downstream_datasets/Hao_PBMC/Hao_PBMC.h5ad')
OUTPUT_ROOT = Path('/workspace/results/analysis/Hao_PBMC/pretraining_pca')
HAO_EMBEDDINGS = HAO_H5AD.parent / 'embeddings' / 'pca_pretraining_2048.npy'
FROM_SCRATCH = False  # True deletes ALL this script's runs, including smoke runs.

N_GENES = 60_000  # Same leading columns, in the same order, as scUNVEIL training.
N_COMPONENTS = 2_048
LIMIT_N_TRAIN_CELLS = None  # E.g. 4096; takes the beginning of the shuffled corpus.
LIMIT_N_HAO_CELLS = None  # Optional projection smoke test.
BATCH_SIZE = 50_000
COVARIANCE_DEVICE = 'cuda:0'  # 'cpu' is also supported for small validation tests.
GPU_PRODUCT_DTYPE = 'float32'  # Or 'float32' for faster, explicitly lower-precision products.
COVARIANCE_GENE_BLOCK = 1_000  # Bounds GPU result tiles and CPU accumulation copies.
CHECKPOINT_EVERY_N_CELLS = 1_000_000  # Covariance checkpoint is ~28.8 GB at 60k.
N_THREADS = 16
EIGEN_TOL = 1e-8  # Legacy identity field: retain to reuse covariance checkpoints.
PCA_N_OVERSAMPLES = 64
PCA_N_ITER = 3
SEED = 42
STAGES = ('mean', 'covariance', 'pca', 'hao')  # Can run stages separately, in order.
PUBLISH_HAO_EMBEDDINGS = True  # Only full 60k -> 2048, unrestricted runs publish.
# -------------------------------------------------------------------------

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


ALGORITHM = 'full-corpus-log1p-centered-dense-upper-covariance-v2'
STOP_REQUESTED = False


class Stopped(Exception):
    pass


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    tqdm.write('Stop requested: finishing the current batch and saving progress.')


def progress(total, desc, initial=0, unit='cells', leave=True):
    return tqdm(total=total, initial=initial, desc=desc, unit=unit,
                smoothing=0, mininterval=1, leave=leave)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def atomic_file(path):
    """A completed file appears only after all its bytes are durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    try:
        with temporary.open('wb') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    with atomic_file(path) as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def save_array(path, value):
    with atomic_file(path) as stream:
        np.save(stream, value, allow_pickle=False)


def save_archive(path, **values):
    with atomic_file(path) as stream:
        np.savez(stream, **values)


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def read_strings(group, name):
    return np.asarray(ad.io.read_elem(group[name])).astype(str)


def normalized_id(value):
    return re.sub(r'^(ENSG\d+)\.\d+$', r'\1', str(value).strip())


@contextlib.contextmanager
def script_lock():
    """Keep the reset lock outside OUTPUT_ROOT so deletion cannot unlink it."""
    parent = OUTPUT_ROOT.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f'.{OUTPUT_ROOT.name}.script.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another instance of this PCA script is running.') from None
        yield


def reset_artifacts():
    """Delete only recognized PCA artifacts, refusing any active legacy run."""
    root = OUTPUT_ROOT.resolve()
    protected = {Path('/'), Path('/workspace'), TRAIN_SHARDS.resolve(),
                 TRAIN_SHARDS.resolve().parent, HAO_H5AD.resolve().parent,
                 HAO_EMBEDDINGS.resolve().parent}
    if root in protected or any(root in source.resolve().parents
                                for source in (TRAIN_SHARDS, REFERENCE_VAR, HAO_H5AD)):
        raise ValueError(f'Unsafe OUTPUT_ROOT for FROM_SCRATCH: {root}')
    runs = []
    with contextlib.ExitStack() as locks:
        for child in sorted(root.iterdir()) if root.exists() else []:
            if not child.is_dir() or child.is_symlink() or not re.fullmatch(
                    r'(full|smoke-\d+)-\d+genes-\d+pcs-[0-9a-f]{12}', child.name):
                raise ValueError(f'Refusing to delete unrecognized artifact: {child}')
            manifest = child / 'manifest.json'
            if manifest.exists() and not json.loads(manifest.read_text()).get(
                    'algorithm', '').startswith('full-corpus-log1p-centered-'):
                raise ValueError(f'Refusing to delete a foreign run: {child}')
            stream = locks.enter_context((child / '.lock').open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Cannot reset: a PCA process is using {child}') from None
            runs.append(child)
        # Check published provenance before deleting anything.
        provenance = HAO_EMBEDDINGS.with_suffix('.provenance.json')
        if provenance.exists():
            fit = Path(json.loads(provenance.read_text())['fit_directory']).resolve()
            if root not in fit.parents:
                raise ValueError('Published Hao embeddings belong to a different OUTPUT_ROOT.')
        elif HAO_EMBEDDINGS.exists():
            raise ValueError('Refusing to delete published embeddings without PCA provenance.')
        for child in runs:
            print(f'FROM_SCRATCH: deleting {child}', flush=True)
            shutil.rmtree(child)
        for path in (HAO_EMBEDDINGS, provenance,
                     HAO_EMBEDDINGS.with_name(HAO_EMBEDDINGS.name + '.partial'),
                     provenance.with_name(provenance.name + '.partial')):
            path.unlink(missing_ok=True)
    print('FROM_SCRATCH: previous PCA artifacts removed.', flush=True)


def prepare():
    if not 0 < N_COMPONENTS < N_GENES:
        raise ValueError('Require 0 < N_COMPONENTS < N_GENES.')
    if any(value <= 0 for value in (BATCH_SIZE, COVARIANCE_GENE_BLOCK,
                                    CHECKPOINT_EVERY_N_CELLS, N_THREADS)):
        raise ValueError('Batch, gene-block, checkpoint and thread sizes must be positive.')
    if not set(STAGES) <= {'mean', 'covariance', 'pca', 'hao'}:
        raise ValueError('Unknown stage in STAGES.')
    if GPU_PRODUCT_DTYPE not in ('float32', 'float64'):
        raise ValueError("GPU_PRODUCT_DTYPE must be 'float32' or 'float64'.")
    for limit in (LIMIT_N_TRAIN_CELLS, LIMIT_N_HAO_CELLS):
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError('Cell limits must be positive integers or None.')
    reference = pd.read_csv(REFERENCE_VAR).iloc[:N_GENES]
    if len(reference) != N_GENES:
        raise ValueError('Reference table has fewer than N_GENES rows.')
    genes = np.array([normalized_id(v) for v in reference['feature_id']])
    if len(set(genes)) != len(genes):
        raise ValueError('Reference gene IDs must be unique.')
    shards, total = [], 0
    paths = sorted(TRAIN_SHARDS.glob('*.h5ad'))
    if not paths:
        raise FileNotFoundError(f'No training shards in {TRAIN_SHARDS}')
    for path in tqdm(paths, desc='Inspect training shards', unit='shards',
                     smoothing=0, mininterval=1):
        with h5py.File(path, 'r') as file:
            shape = tuple(map(int, file['X'].attrs['shape']))
            if file['X'].attrs['encoding-type'] != 'csr_matrix' or shape[1] < N_GENES:
                raise ValueError(f'{path}: expected sparse CSR counts with >= {N_GENES} genes.')
        take = shape[0] if LIMIT_N_TRAIN_CELLS is None else min(
            shape[0], LIMIT_N_TRAIN_CELLS - total)
        shards.append({**fingerprint(path), 'n_cells': int(take)})
        total += take
        if LIMIT_N_TRAIN_CELLS is not None and total >= LIMIT_N_TRAIN_CELLS:
            break
    if LIMIT_N_TRAIN_CELLS is not None and total < LIMIT_N_TRAIN_CELLS:
        raise ValueError('LIMIT_N_TRAIN_CELLS exceeds available training cells.')
    if total <= N_COMPONENTS:
        raise ValueError('Need more training cells than N_COMPONENTS (also for smoke tests).')
    identity = {'algorithm': ALGORITHM, 'n_genes': N_GENES, 'n_components': N_COMPONENTS,
                'product_dtype': GPU_PRODUCT_DTYPE, 'accumulation_dtype': 'float64',
                'tf32': False,
                'limit_n_train_cells': LIMIT_N_TRAIN_CELLS,
                'eigen_tol': EIGEN_TOL, 'seed': SEED,
                'reference': fingerprint(REFERENCE_VAR),
                'gene_ids_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
                'n_train_cells': total, 'shards': shards}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    scope = 'full' if LIMIT_N_TRAIN_CELLS is None else f'smoke-{total}'
    root = OUTPUT_ROOT / f'{scope}-{N_GENES}genes-{N_COMPONENTS}pcs-{digest}'
    root.mkdir(parents=True, exist_ok=True)
    return root, identity, genes


@contextlib.contextmanager
def lock_run(root):
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another PCA process is using {root}') from None
        # Incomplete atomic writes from a killed process are never checkpoints.
        for path in root.rglob('*.partial'):
            path.unlink()
        yield


@contextlib.contextmanager
def lock_other_runs(root):
    """Do not allocate a second 28.8 GB covariance beside an active legacy run."""
    with contextlib.ExitStack() as locks:
        for path in sorted(OUTPUT_ROOT.glob('*/.lock')):
            if path.parent == root:
                continue
            stream = locks.enter_context(path.open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Another PCA run is active in {path.parent}. '
                                   'Stop that run before starting a second covariance fit.') from None
        yield


def count_batches(identity, genes, start=0):
    """Stream actual cropped training counts; no training-generator dilution."""
    offset = 0
    for shard in identity['shards']:
        n = shard['n_cells']
        if start >= offset + n:
            offset += n
            continue
        with h5py.File(shard['path'], 'r') as file:
            observed = np.array([normalized_id(v) for v in read_strings(file['var'], 'feature_id')[:N_GENES]])
            if not np.array_equal(observed, genes):
                raise ValueError(f"Training gene order differs: {shard['path']}")
            x = file['X']
            n_columns = int(x.attrs['shape'][1])
            for begin in range(max(0, start - offset), n, BATCH_SIZE):
                end = min(begin + BATCH_SIZE, n)
                pointers = np.asarray(x['indptr'][begin:end + 1], dtype=np.int64)
                left, right = int(pointers[0]), int(pointers[-1])
                block = sparse.csr_matrix((np.asarray(x['data'][left:right], dtype=np.float64),
                                           x['indices'][left:right], pointers - left),
                                          shape=(end - begin, n_columns))[:, :N_GENES]
                if not np.isfinite(block.data).all() or (block.data < 0).any() or not np.equal(
                        block.data, np.floor(block.data)).all():
                    raise ValueError('Training X must contain finite, nonnegative integer counts.')
                block.sum_duplicates()
                np.log1p(block.data, out=block.data)
                yield offset + end, block
        offset += n


def means(root, identity, genes):
    result = root / 'mean.npz'
    if result.exists():
        print('Mean: already complete.')
        return
    checkpoint = root / 'mean_checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            done, sums = int(saved['n_cells']), saved['sums']
    else:
        done, sums = 0, np.zeros(N_GENES, dtype=np.float64)
    last = done
    with progress(identity['n_train_cells'], 'Mean scan', done) as bar:
        for end, block in count_batches(identity, genes, done):
            sums += np.asarray(block.sum(axis=0)).ravel()
            bar.update(end - done)
            done = end
            if done - last >= CHECKPOINT_EVERY_N_CELLS or STOP_REQUESTED:
                save_archive(checkpoint, sums=sums, n_cells=done)
                last = done
            if STOP_REQUESTED:
                raise Stopped
    save_archive(result, mean=sums / done, n_cells=done, gene_ids=genes)
    checkpoint.unlink(missing_ok=True)


def save_cross_checkpoint(root, matrix, done):
    path = root / f'cross_products_{done:012d}.npy'
    print(f'Checkpointing cross-products at {done:,} cells ({matrix.nbytes / 1e9:.1f} GB).', flush=True)
    with progress(1, 'Write covariance checkpoint', unit='file') as bar:
        save_array(path, matrix)
        bar.update(1)
    for previous in root.glob('cross_products_*.npy'):
        if previous != path:
            previous.unlink()


class DenseCrossProducts:
    """Upload one dense batch, compute upper row blocks, accumulate on the CPU."""

    def __init__(self):
        import torch

        self.torch = torch
        self.device = torch.device(COVARIANCE_DEVICE)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('COVARIANCE_DEVICE must be cpu or cuda:<index>.')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; choose a CUDA environment or '
                               "set COVARIANCE_DEVICE='cpu' for a small smoke test.")
        self.numpy_dtype = np.dtype(GPU_PRODUCT_DTYPE)
        torch.set_num_threads(N_THREADS)
        # Do not silently round FP32 operands to TF32's shorter mantissa.
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
        else:
            torch.set_float32_matmul_precision('highest')
        print(f'Dense cross-products: {self.device}, {GPU_PRODUCT_DTYPE} multiplication; '
              'float64 CPU accumulation; upper triangle only; TF32 disabled.', flush=True)

    def add(self, products, block):
        torch = self.torch
        # The complete 60k-square covariance stays on the CPU. Only this dense
        # cell batch and one gene-row product tile need to fit on the GPU.
        input_bytes = block.shape[0] * N_GENES * self.numpy_dtype.itemsize
        tile_bytes = min(COVARIANCE_GENE_BLOCK, N_GENES) * N_GENES * self.numpy_dtype.itemsize
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            # Include cached allocations reusable by this process, but reserve
            # headroom for cuBLAS workspace and possible operand packing.
            reusable = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            required = input_bytes + 2 * tile_bytes + 512 * 1024**2
            if required > free + reusable:
                raise RuntimeError(f'GPU batch needs approximately {required / 1e9:.2f} GB; '
                                   f'{(free + reusable) / 1e9:.2f} GB available. '
                                   'Reduce BATCH_SIZE or COVARIANCE_GENE_BLOCK and rerun to resume.')
        with torch.inference_mode():
            # Fortran layout keeps each gene's cell vector contiguous, avoiding
            # repeated packing of the full batch for the successive GEMMs.
            host = block.astype(self.numpy_dtype, copy=False).toarray(order='F')
            device_batch = torch.from_numpy(host).to(self.device)
            del host
            try:
                with progress(N_GENES, 'Current dense GPU batch' if self.device.type == 'cuda'
                              else 'Current dense CPU batch', unit='genes', leave=False) as tiles:
                    for begin in range(0, N_GENES, COVARIANCE_GENE_BLOCK):
                        end = min(begin + COVARIANCE_GENE_BLOCK, N_GENES)
                        # Skip cross-block lower triangles. The diagonal block
                        # is computed in full; its upper half is authoritative.
                        tile = torch.mm(device_batch[:, begin:end].T, device_batch[:, begin:])
                        cpu_tile = tile.cpu().numpy()
                        products[begin:end, begin:] += cpu_tile
                        del cpu_tile, tile
                        tiles.update(end - begin)
            finally:
                del device_batch

    def close(self):
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()


def covariance(root, identity, genes):
    result = root / 'covariance.npy'
    if result.exists():
        print('Covariance: already complete.')
        return
    with np.load(root / 'mean.npz') as saved:
        mean = saved['mean']
    checkpoints = sorted(root.glob('cross_products_*.npy'))
    if checkpoints:
        latest = checkpoints[-1]
        done = int(latest.stem.split('_')[-1])
        products = np.load(latest)  # One writable float64 covariance-sized allocation.
        if products.shape != (N_GENES, N_GENES) or products.dtype != np.float64:
            raise ValueError('Invalid cross-product checkpoint.')
        for previous in checkpoints[:-1]:
            previous.unlink()
    else:
        done = 0
        products = np.zeros((N_GENES, N_GENES), dtype=np.float64)
    last = done
    engine = DenseCrossProducts()
    try:
        with progress(identity['n_train_cells'], 'Cross-product scan', done) as bar:
            for end, block in count_batches(identity, genes, done):
                engine.add(products, block)
                bar.update(end - done)
                done = end  # A checkpoint includes whole batches, exactly once.
                if done - last >= CHECKPOINT_EVERY_N_CELLS or STOP_REQUESTED:
                    save_cross_checkpoint(root, products, done)
                    last = done
                if STOP_REQUESTED:
                    raise Stopped
    finally:
        engine.close()
    if last != done:
        save_cross_checkpoint(root, products, done)
    n = identity['n_train_cells']
    with progress(N_GENES, 'Mirror covariance', unit='genes') as bar:
        for begin in range(0, N_GENES, COVARIANCE_GENE_BLOCK):
            end = min(begin + COVARIANCE_GENE_BLOCK, N_GENES)
            products[begin:end, :begin] = products[:begin, begin:end].T
            diagonal = products[begin:end, begin:end]
            lower = np.tril_indices(end - begin, -1)
            diagonal[lower] = diagonal.T[lower]
            bar.update(end - begin)
            if STOP_REQUESTED:
                raise Stopped
    # In-place centering avoids a second full-sized allocation. The uncentered
    # checkpoint remains intact until the completed covariance is durable.
    with progress(N_GENES, 'Center covariance', unit='genes') as bar:
        for begin in range(0, N_GENES, COVARIANCE_GENE_BLOCK):
            end = min(begin + COVARIANCE_GENE_BLOCK, N_GENES)
            products[begin:end] -= np.multiply.outer(mean[begin:end], mean) * n
            products[begin:end] /= n - 1
            bar.update(end - begin)
            if STOP_REQUESTED:
                raise Stopped
    save_array(result, products)
    for checkpoint in root.glob('cross_products_*.npy'):
        checkpoint.unlink()


def principal_components(root, identity, genes):
    result = root / 'pca.npz'
    if result.exists():
        print('PCA: already complete.')
        return
    cov = np.load(root / 'covariance.npy', mmap_mode='r')
    with progress(1, 'Randomized PCA (library-call ETA unavailable)', unit='solve') as bar:
        _, _, components = randomized_svd(
            cov, n_components=N_COMPONENTS, n_oversamples=PCA_N_OVERSAMPLES,
            n_iter=PCA_N_ITER, random_state=SEED)
        bar.update(1)
    if STOP_REQUESTED:
        raise Stopped
    vectors = components.T
    # Diagonalize covariance within the approximate subspace so saved variances
    # describe the actual returned directions, not singular values of covariance.
    cov_vectors = np.empty_like(vectors)
    with progress(N_GENES, 'Check PCA covariance projection', unit='genes') as bar:
        for begin in range(0, N_GENES, COVARIANCE_GENE_BLOCK):
            end = min(begin + COVARIANCE_GENE_BLOCK, N_GENES)
            cov_vectors[begin:end] = cov[begin:end] @ vectors
            bar.update(end - begin)
            if STOP_REQUESTED:
                raise Stopped
    reduced = vectors.T @ cov_vectors
    eigenvalues, rotation = linalg.eigh((reduced + reduced.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, rotation = eigenvalues[order], rotation[:, order]
    vectors, cov_vectors = vectors @ rotation, cov_vectors @ rotation
    if eigenvalues[-1] <= np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1):
        raise ValueError('Insufficient positive PCA rank; use more smoke-test cells/genes.')
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(N_COMPONENTS)])
    vectors *= signs[None, :]
    cov_vectors *= signs[None, :]
    residual_ss = np.zeros(N_COMPONENTS)
    with progress(N_GENES, 'Check PCA residuals', unit='genes') as bar:
        for begin in range(0, N_GENES, COVARIANCE_GENE_BLOCK):
            end = min(begin + COVARIANCE_GENE_BLOCK, N_GENES)
            residual = cov_vectors[begin:end] - vectors[begin:end] * eigenvalues
            residual_ss += np.sum(residual * residual, axis=0)
            bar.update(end - begin)
            if STOP_REQUESTED:
                raise Stopped
    relative_residuals = np.sqrt(residual_ss) / eigenvalues
    orthogonality_error = float(np.max(np.abs(vectors.T @ vectors - np.eye(N_COMPONENTS))))
    if (not np.isfinite(eigenvalues).all() or not np.isfinite(relative_residuals).all()
            or not np.isfinite(orthogonality_error) or orthogonality_error > 1e-7):
        raise ValueError('PCA finite-value/orthogonality check failed; no basis published.')
    with np.load(root / 'mean.npz') as saved:
        mean = saved['mean']
    trace = float(np.trace(cov))
    captured = float(eigenvalues.sum())
    if not np.isfinite(trace) or trace <= 0 or not 0 < captured <= trace * (1 + 1e-7):
        raise ValueError('Invalid PCA captured variance; no basis published.')
    save_archive(result, mean=mean, components=vectors.T.copy(), eigenvalues=eigenvalues,
                 explained_variance_ratio=eigenvalues / trace, gene_ids=genes,
                 n_train_cells=identity['n_train_cells'], relative_residuals=relative_residuals,
                 orthogonality_error=orthogonality_error,
                 solver='sklearn.randomized_svd + projected-covariance eigh',
                 n_oversamples=PCA_N_OVERSAMPLES, n_iter=PCA_N_ITER, seed=SEED)
    print(f'PCA complete: explained variance {eigenvalues.sum() / trace:.4%}; '
          f'max relative residual {relative_residuals.max():.3g}.')


def project_hao(root, genes):
    source_identity = {'source': fingerprint(HAO_H5AD), 'limit_n_hao_cells': LIMIT_N_HAO_CELLS}
    digest = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()[:12]
    output = root / f'hao-{digest}'
    output.mkdir(exist_ok=True)
    result = output / 'embeddings.npy'
    if not result.exists():
        with np.load(root / 'pca.npz') as saved:
            mean, components = saved['mean'], saved['components']
        offset = mean @ components.T
        with h5py.File(HAO_H5AD, 'r') as file:
            if 'raw' not in file:
                raise ValueError('Hao must provide integer UMI counts in raw.X.')
            raw = file['raw']
            x = raw['X']
            if x.attrs['encoding-type'] != 'csr_matrix':
                raise ValueError('Expected CSR raw counts in Hao.')
            n_cells, n_features = map(int, x.attrs['shape'])
            if LIMIT_N_HAO_CELLS is not None:
                if LIMIT_N_HAO_CELLS > n_cells:
                    raise ValueError('LIMIT_N_HAO_CELLS exceeds available cells.')
                n_cells = LIMIT_N_HAO_CELLS
            var = raw['var']
            identifiers = read_strings(var, 'feature_id' if 'feature_id' in var else var.attrs['_index'])
            lookup = {gene: i for i, gene in enumerate(genes)}
            mapped = np.array([lookup.get(normalized_id(gene), -1) for gene in identifiers])
            selected = np.flatnonzero(mapped >= 0)
            if not len(selected) or len(np.unique(mapped[selected])) != len(selected):
                raise ValueError('Hao gene mapping is empty or has duplicate target genes.')
            mapping = sparse.csr_matrix((np.ones(len(selected)), (selected, mapped[selected])),
                                        shape=(n_features, N_GENES))
            print(f'Hao: mapping {len(selected):,}/{n_features:,} genes by Ensembl ID.')
            checkpoint = output / 'progress.json'
            work = output / 'embeddings.incomplete.npy'
            if checkpoint.exists():
                done = json.loads(checkpoint.read_text())['n_cells']
                embeddings = np.lib.format.open_memmap(work, mode='r+')
                if embeddings.shape != (n_cells, N_COMPONENTS) or embeddings.dtype != np.float32:
                    raise ValueError('Invalid Hao projection checkpoint.')
            else:
                done = 0
                embeddings = np.lib.format.open_memmap(work, mode='w+', dtype=np.float32,
                                                       shape=(n_cells, N_COMPONENTS))
            with progress(n_cells, 'Project Hao', done) as bar:
                for begin in range(done, n_cells, BATCH_SIZE):
                    end = min(begin + BATCH_SIZE, n_cells)
                    pointers = np.asarray(x['indptr'][begin:end + 1], dtype=np.int64)
                    left, right = int(pointers[0]), int(pointers[-1])
                    block = sparse.csr_matrix((np.asarray(x['data'][left:right], dtype=np.float64),
                                               x['indices'][left:right], pointers - left),
                                              shape=(end - begin, n_features))
                    if not np.isfinite(block.data).all() or (block.data < 0).any() or not np.equal(
                            block.data, np.floor(block.data)).all():
                        raise ValueError('Hao raw.X must contain nonnegative integer UMI counts.')
                    block = block @ mapping
                    np.log1p(block.data, out=block.data)
                    projected = block @ components.T - offset
                    if not np.isfinite(projected).all():
                        raise ValueError('Nonfinite Hao PCA coordinates.')
                    embeddings[begin:end] = projected
                    embeddings.flush()
                    with work.open('rb') as stream:
                        os.fsync(stream.fileno())
                    save_json(checkpoint, {'n_cells': end})
                    bar.update(end - begin)
                    if STOP_REQUESTED:
                        raise Stopped
            del embeddings
        save_json(output / 'source.json', {**source_identity, 'n_cells': n_cells,
                                          'mapped_genes': len(selected)})
        os.replace(work, result)
        fsync_directory(output)
        checkpoint.unlink(missing_ok=True)
    full_run = (LIMIT_N_TRAIN_CELLS is None and LIMIT_N_HAO_CELLS is None
                and N_GENES == 60_000 and N_COMPONENTS == 2_048)
    if PUBLISH_HAO_EMBEDDINGS and full_run:
        with atomic_file(HAO_EMBEDDINGS) as stream, result.open('rb') as source:
            shutil.copyfileobj(source, stream, length=16 * 1024**2)
        save_json(HAO_EMBEDDINGS.with_suffix('.provenance.json'),
                  {'fit_directory': str(root), 'projection': str(result),
                   'training': json.loads((root / 'manifest.json').read_text()),
                   'hao_source': source_identity})
        print(f'Hao representation: {HAO_EMBEDDINGS}')
    else:
        print(f'Hao representation (not published to benchmark inputs): {result}')


def main():
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with script_lock():
        if FROM_SCRATCH:
            reset_artifacts()
        root, identity, genes = prepare()
        print(f'Artifacts: {root}\nTraining cells: {identity["n_train_cells"]:,}\n'
              f'Float64 covariance: {N_GENES**2 * 8 / 1e9:.2f} GB', flush=True)
        with lock_other_runs(root), lock_run(root), threadpool_limits(limits=N_THREADS):
            save_json(root / 'manifest.json', identity)
            for stage in STAGES:
                if STOP_REQUESTED:
                    raise Stopped
                start = time.perf_counter()
                if stage == 'mean':
                    means(root, identity, genes)
                elif stage == 'covariance':
                    covariance(root, identity, genes)
                elif stage == 'pca':
                    principal_components(root, identity, genes)
                else:
                    project_hao(root, genes)
                print(f'{stage}: finished in {time.perf_counter() - start:.1f} s.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Stopped:
        print('Stopped safely. Run the same script again to resume.', flush=True)
        raise SystemExit(130)
