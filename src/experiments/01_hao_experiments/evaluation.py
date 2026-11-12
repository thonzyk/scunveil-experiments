import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def _labels(obs, key, n_samples):
    values = np.asarray(obs[key])
    if len(values) != n_samples:
        raise ValueError(f'obs[{key!r}] length must match X rows')
    return np.unique(values, return_inverse=True)[1]


def _groups(obs, key, n_samples, random_state):
    values = np.asarray(obs[key])
    if len(values) != n_samples:
        raise ValueError(f'obs[{key!r}] length must match X rows')

    unique = np.unique(values)
    if len(unique) < 2:
        raise ValueError('At least two groups are required')

    return values, np.random.default_rng(random_state).permutation(unique)


def train_test_split(X, obs, lbl_key, split_key, test_ratio=0.5, random_state=None):
    if not 0 < test_ratio < 1:
        raise ValueError('test_ratio must be between 0 and 1')

    y = _labels(obs, lbl_key, X.shape[0])
    groups, shuffled = _groups(obs, split_key, X.shape[0], random_state)
    n_test = int(np.clip(round(len(shuffled) * test_ratio), 1, len(shuffled) - 1))
    is_test = np.isin(groups, shuffled[:n_test])

    return X[~is_test], y[~is_test], X[is_test], y[is_test]


def train_dev_test_split(X, obs, lbl_key, split_key, dev_ratio=0.5, random_state=None):
    if not 0 <= dev_ratio <= 1:
        raise ValueError('dev_ratio must be between 0 and 1')

    y = _labels(obs, lbl_key, X.shape[0])
    groups, shuffled = _groups(obs, split_key, X.shape[0], random_state)
    test_group, remaining = shuffled[:1], shuffled[1:]
    n_dev = int(np.floor(len(remaining) * dev_ratio))

    is_test = np.isin(groups, test_group)
    is_dev = np.isin(groups, remaining[:n_dev])
    is_train = ~(is_test | is_dev)

    return X[is_train], y[is_train], X[is_dev], y[is_dev], X[is_test], y[is_test]


def init_linear_model(input_size, n_classes, learning_rate):
    model = Sequential([
        Input(shape=(input_size,)),
        Dense(n_classes, kernel_initializer='zeros'),
    ])
    model.compile(
        optimizer=Adam(learning_rate),
        loss=SparseCategoricalCrossentropy(from_logits=True, name='xent'),
    )
    return model


# Compatibility with older benchmark scripts.
test_split = train_test_split