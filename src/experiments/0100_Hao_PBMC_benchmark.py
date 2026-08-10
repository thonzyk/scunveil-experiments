import csv
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from constants import ANALYSIS_ROOT, DATA_ROOT
from evaluation import init_linear_model, train_dev_test_split, train_test_split


BATCH_SIZE = 1024
SWEEP_EPOCHS = 1000
FINAL_EPOCHS = SWEEP_EPOCHS
PATIENCE = 50
LEARNING_RATES = np.logspace(1, -7, 17)
# LEARNING_RATES = [1e-4]

METHODS = (
    # Embeddings
    'scunveil_emb_2048',
    'scunveil_emb_512',
    'scunveil_emb_128',
    'scunveil_emb_50',
    'scunveil_emb_10',
    'scunveil_linear_emb_2048',

    'harmony_pca_10',
    'harmony_pca_50',
    'harmony_pca_512',
    'scfoundation',
    'scgpt',
    'scimilarity',
    'scvi_emb_10',
    'scvi_emb_50',
    'scvi_emb_512',
    'uce_4l_1280',
    'uce_33l_1280',

    # Imputed marker representations
    'magic_50_markers',
    'scunveil_50_markers',
    'scvi_50_markers',

    # Raw measurements
    'rna_log1p',
    'rna_marker_log1p',
)

REPRESENTATIONS = {
    # Embeddings
    'scunveil_emb_2048': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_2048.npy', None),
    'scunveil_emb_512': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_2048.npy', 512),
    'scunveil_emb_128': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_2048.npy', 128),
    'scunveil_emb_50': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_2048.npy', 50),
    'scunveil_emb_10': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_2048.npy', 10),
    'scunveil_linear_emb_2048': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scunveil_linear_2048.npy', None),

    'harmony_pca_10': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/harmony_pca_10.npy', None),
    'harmony_pca_50': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/harmony_pca_50.npy', None),
    'harmony_pca_512': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/harmony_pca_512.npy', None),
    'scfoundation': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scFoundation.npy', None),
    'scgpt': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scGPT.npy', None),
    'scimilarity': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/SCimilarity.npy', None),
    'scvi_emb_10': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scvi_10.npy', None),
    'scvi_emb_50': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scvi_50.npy', None),
    'scvi_emb_512': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/scvi_512.npy', None),
    'uce_4l_1280': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/UCE_4L_1280.npy', None),
    'uce_33l_1280': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/embeddings/UCE_33L_1280.npy', None),

    # Imputation
    'magic_50_markers': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/imputation/magic_50_marker.npy', None),
    'scunveil_50_markers': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/imputation/scunveil_50_markers.npy', None),
    'scvi_50_markers': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/imputation/scvi_50_markers.npy', None),

    # Raw measurements
    'rna_log1p': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/raw_measurement/rna_log1p.npz', None),
    'rna_marker_log1p': (DATA_ROOT / 'downstream_datasets/Hao_PBMC/raw_measurement/rna_marker_log1p.npy', None),
}

TEMP_ROOT = Path('_temp')
DATASET_PATH = DATA_ROOT / 'downstream_datasets/Hao_PBMC/Hao_PBMC.h5ad'
HAO_ROOT = ANALYSIS_ROOT / 'Hao_PBMC'
HAO_ROOT.mkdir(parents=True, exist_ok=True)

LR_SWEEP_PATH = HAO_ROOT / 'cell_typing_lr_sweep.tsv'
CELL_TYPING_PATH = HAO_ROOT / 'cell_typing.tsv'

LR_SWEEP_FIELDS = (
    'representation_type', 'learning_rate', 'best_epoch', 'best_test_loss',
    'train_loss_at_best_epoch', 'accuracy', 'balanced_accuracy', 'macro_f1',
)
CELL_TYPING_FIELDS = (
    'representation_type', 'accuracy', 'balanced_accuracy', 'macro_f1', 'early_stop',
)


class BestValidationCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss = np.inf
        self.train_loss = np.inf
        self.epoch = None
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None and loss < self.loss:
            self.loss = float(loss)
            self.train_loss = float(logs['loss'])
            self.epoch = epoch + 1
            self.weights = self.model.get_weights()


def format_lr(value):
    return f'{float(value):.3e}'


def ensure_tsv(path, fields):
    if path.exists():
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        csv.DictWriter(file, fieldnames=fields, delimiter='\t', lineterminator='\n').writeheader()


def read_tsv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file, delimiter='\t'))


def append_tsv(path, fields, row):
    with path.open('a', newline='', encoding='utf-8') as file:
        csv.DictWriter(file, fieldnames=fields, delimiter='\t', lineterminator='\n').writerow(row)


def load_representation(method):
    try:
        filename, n_features = REPRESENTATIONS[method]
    except KeyError as error:
        raise ValueError(f'Unknown representation: {method}') from error

    path = TEMP_ROOT / filename
    X = load_npz(path) if path.suffix == '.npz' else np.load(path)
    return X if n_features is None else X[:, :n_features]


def dense_float32(X):
    X = X.toarray() if sp.issparse(X) else X
    return np.asarray(X, dtype=np.float32)


def load_train_test(method, random_state=None):
    adata = ad.read_h5ad(DATASET_PATH, backed='r')
    x_train, y_train, x_test, y_test = train_test_split(
        load_representation(method),
        adata.obs,
        lbl_key='celltype.l3',
        split_key='donor_id',
        test_ratio=0.5,
        random_state=random_state,
    )

    x_train, x_test = dense_float32(x_train), dense_float32(x_test)
    n_classes = len(np.unique(np.concatenate((y_train, y_test))))
    return x_train, y_train, x_test, y_test, x_train.shape[1], n_classes


def load_train_dev_test(method, random_state=None):
    adata = ad.read_h5ad(DATASET_PATH, backed='r')
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        load_representation(method),
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


def fit_probe(x_train, y_train, x_val, y_val, n_features, n_classes, lr, epochs):
    tf.keras.backend.clear_session()
    model = init_linear_model(n_features, n_classes, lr)
    best = BestValidationCheckpoint()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
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
    predicted = np.argmax(model.predict(X, batch_size=BATCH_SIZE, verbose=0), axis=1)
    return {
        'accuracy': accuracy_score(y, predicted),
        'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'macro_f1': f1_score(y, predicted, average='macro'),
    }


def completed_sweep_jobs():
    completed = set()
    for row in read_tsv(LR_SWEEP_PATH):
        try:
            completed.add((row['representation_type'], format_lr(row['learning_rate'])))
        except (KeyError, TypeError, ValueError):
            pass
    return completed


def next_sweep_job():
    completed = completed_sweep_jobs()
    for method in METHODS:
        for lr in LEARNING_RATES:
            if (method, format_lr(lr)) not in completed:
                return method, float(lr)
    return None


def best_learning_rates():
    best = {}
    for row in read_tsv(LR_SWEEP_PATH):
        method = row.get('representation_type')
        if method not in METHODS:
            continue
        try:
            candidate = float(row['learning_rate']), float(row['best_test_loss'])
        except (KeyError, TypeError, ValueError):
            continue
        if method not in best or candidate[1] < best[method][1]:
            best[method] = candidate

    missing = set(METHODS) - set(best)
    if missing:
        raise RuntimeError(f'Missing learning-rate sweep results: {sorted(missing)}')
    return {method: best[method][0] for method in METHODS}


def print_sweep_summary():
    rows = read_tsv(LR_SWEEP_PATH)

    print()
    print('=' * 80)
    print('LEARNING-RATE SWEEP SUMMARY')
    print('=' * 80)

    for method in METHODS:
        method_rows = []

        for row in rows:
            if row.get('representation_type') != method:
                continue

            try:
                method_rows.append({
                    'learning_rate': float(row['learning_rate']),
                    'best_epoch': int(row['best_epoch']),
                    'best_test_loss': float(row['best_test_loss']),
                    'accuracy': float(row['accuracy']),
                    'balanced_accuracy': float(row['balanced_accuracy']),
                    'macro_f1': float(row['macro_f1']),
                })
            except (KeyError, TypeError, ValueError):
                continue

        if not method_rows:
            print(f'{method}: no valid sweep results')
            continue

        selected = min(method_rows, key=lambda row: row['best_test_loss'])

        lr = selected['learning_rate']
        best_epoch = selected['best_epoch']

        print(
            f'{method}: '
            f'lr={format_lr(lr)}, '
            f'best_epoch={best_epoch}, '
            f'val_loss={selected["best_test_loss"]:.6g}, '
            f'accuracy={selected["accuracy"]:.4f}, '
            f'balanced_accuracy={selected["balanced_accuracy"]:.4f}, '
            f'macro_f1={selected["macro_f1"]:.4f}'
        )

        if best_epoch >= SWEEP_EPOCHS:
            print(
                f'  WARNING: best epoch reached the sweep limit ({SWEEP_EPOCHS}). '
                'Consider increasing SWEEP_EPOCHS.'
            )

        if best_epoch < 60:
            print(
                f'  WARNING: best epoch was very early ({best_epoch}). '
                'Check whether the learning rate is too large.'
            )

        if np.isclose(lr, np.max(LEARNING_RATES)):
            print(
                '  WARNING: selected learning rate is the largest tested value. '
                'Consider testing larger learning rates.'
            )

        if np.isclose(lr, np.min(LEARNING_RATES)):
            print(
                '  WARNING: selected learning rate is the smallest tested value. '
                'Consider testing smaller learning rates.'
            )

    print('=' * 80)
    print()


def next_cell_typing_method():
    counts = Counter(row.get('representation_type') for row in read_tsv(CELL_TYPING_PATH))
    return min(enumerate(METHODS), key=lambda item: (counts[item[1]], item[0]))[1]


def run_sweep(method, lr):
    print(f'LR sweep: {method}, lr={format_lr(lr)}')
    np.random.seed(42)
    data = load_train_test(method, random_state=42)
    model, best = fit_probe(*data, lr, SWEEP_EPOCHS)

    append_tsv(LR_SWEEP_PATH, LR_SWEEP_FIELDS, {
        'representation_type': method,
        'learning_rate': format_lr(lr),
        'best_epoch': best.epoch,
        'best_test_loss': best.loss,
        'train_loss_at_best_epoch': best.train_loss,
        **metrics(model, data[2], data[3]),
    })

def run_cell_typing(method, lr):
    print(f'Cell typing: {method}, lr={format_lr(lr)}')

    (
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test,
        n_features,
        n_classes,
    ) = load_train_dev_test(method)

    model, best = fit_probe(
        x_train,
        y_train,
        x_dev,
        y_dev,
        n_features,
        n_classes,
        lr,
        FINAL_EPOCHS,
    )

    append_tsv(CELL_TYPING_PATH, CELL_TYPING_FIELDS, {
        'representation_type': method,
        **metrics(model, x_test, y_test),
        'early_stop': best.epoch - 1,
    })


def main():
    ensure_tsv(LR_SWEEP_PATH, LR_SWEEP_FIELDS)
    job = next_sweep_job()

    if job:
        run_sweep(*job)
        return

    print_sweep_summary()

    ensure_tsv(CELL_TYPING_PATH, CELL_TYPING_FIELDS)
    method = next_cell_typing_method()
    run_cell_typing(method, best_learning_rates()[method])


if __name__ == '__main__':
    main()