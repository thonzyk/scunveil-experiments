import csv
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from constants import ANALYSIS_ROOT, DATA_ROOT, RESULTS_ROOT
from evaluation import init_linear_model, train_dev_test_split


EXPERIMENT_ID = '2026-05-26_14-26-04'

BATCH_SIZE = 1024
FINAL_EPOCHS = 1000
PATIENCE = 50
DILUTION_RATES = np.logspace(0, -3, 7)

METHODS = (
    'scunveil_emb_2048',
    'rna_log1p',
)

TEMP_ROOT = Path('_temp')
DATASET_PATH = DATA_ROOT / 'downstream_datasets/Hao_PBMC/Hao_PBMC.h5ad'
REFERENCE_VAR_PATH = DATA_ROOT / 'var_sorted.csv'

EXPERIMENT_ROOT = RESULTS_ROOT / EXPERIMENT_ID
CONFIG_PATH = EXPERIMENT_ROOT / 'config.json'
WEIGHTS_ROOT = EXPERIMENT_ROOT / 'weights'

HAO_ROOT = ANALYSIS_ROOT / 'Hao_PBMC'
HAO_ROOT.mkdir(parents=True, exist_ok=True)

LR_SWEEP_PATH = HAO_ROOT / 'cell_typing_lr_sweep.tsv'
CELL_TYPING_PATH = HAO_ROOT / 'cell_typing_dilution.tsv'

CELL_TYPING_FIELDS = (
    'representation_type',
    'accuracy',
    'balanced_accuracy',
    'macro_f1',
    'early_stop',
    'dilution',
)


class BestValidationCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss = np.inf
        self.epoch = None
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None and loss < self.loss:
            self.loss = float(loss)
            self.epoch = epoch + 1
            self.weights = self.model.get_weights()



def next_dilution_rate(method):
    counts = np.zeros(len(DILUTION_RATES), dtype=int)

    for row in read_tsv(CELL_TYPING_PATH):
        if row.get('representation_type') != method:
            continue

        try:
            recorded_rate = float(row['dilution'])
        except (KeyError, TypeError, ValueError):
            continue

        matches = np.flatnonzero(
            np.isclose(
                DILUTION_RATES,
                recorded_rate,
                rtol=1e-9,
                atol=0.0,
            )
        )
        if matches.size:
            counts[matches[0]] += 1

    return float(DILUTION_RATES[np.argmin(counts)])


def format_lr(value):
    return f'{float(value):.3e}'


def ensure_tsv(path, fields):
    if path.exists():
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        csv.DictWriter(
            file,
            fieldnames=fields,
            delimiter='\t',
            lineterminator='\n',
        ).writeheader()


def read_tsv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file, delimiter='\t'))


def append_tsv(path, fields, row):
    with path.open('a', newline='', encoding='utf-8') as file:
        csv.DictWriter(
            file,
            fieldnames=fields,
            delimiter='\t',
            lineterminator='\n',
        ).writerow(row)


def resolve_input_path(path):
    """Resolve an input from the staging directory first, then its normal path."""
    path = Path(path)
    candidates = (TEMP_ROOT / path, path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'Input path does not exist; checked: {checked}')


def latest_weights_path():
    weights_root = resolve_input_path(WEIGHTS_ROOT)
    candidates = sorted(weights_root.glob('*_weights.h5'))

    if not candidates:
        raise FileNotFoundError(
            f'No *_weights.h5 files found in {weights_root}'
        )

    return candidates[-1]


def dense_float32(X):
    X = X.toarray() if sp.issparse(X) else X
    return np.asarray(X, dtype=np.float32)


def raw_count_adata(adata):
    """Return an AnnData whose X and var describe the raw UMI count layer."""
    if adata.raw is None:
        raise ValueError(
            'Hao PBMC dilution requires raw UMI counts in adata.raw.X'
        )

    counts_adata = adata.raw.to_adata()
    values = (
        counts_adata.X.data
        if sp.issparse(counts_adata.X)
        else np.asarray(counts_adata.X).reshape(-1)
    )

    # Check in bounded chunks so validation does not allocate an additional
    # array proportional to this large dataset's number of observations.
    for start in range(0, values.size, 1_000_000):
        chunk = values[start:start + 1_000_000]
        if not np.all(np.isfinite(chunk)):
            raise ValueError('Raw UMI counts must be finite')
        if np.any(chunk < 0):
            raise ValueError('Raw UMI counts must be non-negative')
        if not np.all(chunk == np.floor(chunk)):
            raise ValueError('Raw UMI counts must be integer-valued')

    return counts_adata


def thin_csr(X, probability):
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f'Dilution probability must be in [0, 1], got {probability}')

    X = X.tocsr(copy=True) if sp.issparse(X) else sp.csr_matrix(X)

    with tf.device('/CPU:0'):
        seed = tf.random.uniform(
            shape=(2,),
            maxval=2**31 - 1,
            dtype=tf.int32,
        )
        X.data = tf.random.stateless_binomial(
            shape=X.data.shape,
            seed=seed,
            counts=tf.cast(X.data, tf.float32),
            probs=float(probability),
            output_dtype=tf.int64,
        ).numpy().astype(X.data.dtype, copy=False)

    X.eliminate_zeros()
    return X


def load_scunveil_embeddings(adata):
    from scunveil import scUnveil
    from constants import SC_UNVEIL_VERSION

    scUNVEIL = scUnveil(model_version=SC_UNVEIL_VERSION)
    scUNVEIL.set_input_anndata(adata, batch_size=1000)

    return scUNVEIL.get_embeddings(n_features=None)


def build_representation(method, adata, diluted_counts):
    if method == 'scunveil_emb_2048':
        adata.X = diluted_counts
        return load_scunveil_embeddings(adata)

    if method == 'rna_log1p':
        return diluted_counts.log1p()

    raise ValueError(f'Unknown representation: {method}')


def load_train_dev_test(method, dilution, random_state=None):
    dataset_path = resolve_input_path(DATASET_PATH)
    adata = raw_count_adata(ad.read_h5ad(dataset_path))

    diluted_counts = thin_csr(adata.X, probability=dilution)
    representation = build_representation(method, adata, diluted_counts)

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        representation,
        adata.obs,
        lbl_key='celltype.l3',
        split_key='donor_id',
        dev_ratio=0.5,
        random_state=random_state,
    )

    x_train = dense_float32(x_train)
    x_dev = dense_float32(x_dev)
    x_test = dense_float32(x_test)

    n_classes = len(np.unique(np.concatenate((y_train, y_dev, y_test))))

    return (
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test,
        x_train.shape[1],
        n_classes,
    )


def fit_probe(x_train, y_train, x_val, y_val, n_features, n_classes, lr):
    tf.keras.backend.clear_session()
    model = init_linear_model(n_features, n_classes, lr)
    best = BestValidationCheckpoint()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=FINAL_EPOCHS,
        shuffle=True,
        callbacks=[
            best,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                mode='min',
                restore_best_weights=False,
            ),
        ],
        verbose=1,
    )

    if best.weights is None:
        raise RuntimeError('Training produced no validation checkpoint')

    model.set_weights(best.weights)
    return model, best


def metrics(model, X, y):
    predicted = np.argmax(
        model.predict(X, batch_size=BATCH_SIZE, verbose=0),
        axis=1,
    )
    return {
        'accuracy': accuracy_score(y, predicted),
        'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'macro_f1': f1_score(y, predicted, average='macro'),
    }


def best_learning_rates():
    best = {}

    for row in read_tsv(LR_SWEEP_PATH):
        method = row.get('representation_type')
        if method not in METHODS:
            continue

        try:
            candidate = (
                float(row['learning_rate']),
                float(row['best_test_loss']),
            )
        except (KeyError, TypeError, ValueError):
            continue

        if method not in best or candidate[1] < best[method][1]:
            best[method] = candidate

    missing = set(METHODS) - set(best)
    if missing:
        raise RuntimeError(
            f'Missing learning-rate sweep results: {sorted(missing)}'
        )

    return {method: best[method][0] for method in METHODS}


def next_cell_typing_method():
    counts = Counter(
        row.get('representation_type')
        for row in read_tsv(CELL_TYPING_PATH)
    )
    return min(
        enumerate(METHODS),
        key=lambda item: (counts[item[1]], item[0]),
    )[1]


def run_cell_typing(method, lr):
    dilution = next_dilution_rate(method)

    print(
        f'Cell typing dilution: {method}, '
        f'lr={format_lr(lr)}, dilution={dilution:.3e}'
    )

    (
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test,
        n_features,
        n_classes,
    ) = load_train_dev_test(method, dilution)

    model, best = fit_probe(
        x_train,
        y_train,
        x_dev,
        y_dev,
        n_features,
        n_classes,
        lr,
    )

    append_tsv(CELL_TYPING_PATH, CELL_TYPING_FIELDS, {
        'representation_type': method,
        **metrics(model, x_test, y_test),
        'early_stop': best.epoch - 1,
        'dilution': dilution,
    })


def main():
    ensure_tsv(CELL_TYPING_PATH, CELL_TYPING_FIELDS)

    method = next_cell_typing_method()
    lr = best_learning_rates()[method]
    run_cell_typing(method, lr)


if __name__ == '__main__':
    main()
