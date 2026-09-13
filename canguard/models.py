"""
The three detectors.

CAN-Guard is deliberately not one model. An in-vehicle IDS has to answer two
different questions, and they need different tools:

  1. "Is this normal?"    -> unsupervised. Trained only on attack-free traffic,
                             so it can flag an attack nobody has seen before.
                             (TensorFlow autoencoder + a Scikit-learn
                             Isolation Forest as the classical baseline.)

  2. "What is it?"        -> supervised. Trained on labelled attacks, so it can
                             name the threat and drive a response.
                             (Scikit-learn Random Forest + a TensorFlow MLP.)

The autoencoder is the part that "establishes a baseline of behaviour": it
learns to rebuild a normal 250 ms window from an 8-number bottleneck. Traffic
it has never seen rebuilds badly, and that reconstruction error is the alarm.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

CLIP = 20.0   # keep wildly out-of-distribution windows numerically sane


def make_scaler(X_clean):
    sc = StandardScaler().fit(X_clean)
    return sc


def scale(sc, X):
    return np.clip(sc.transform(X), -CLIP, CLIP).astype("float32")


# ------------------------------------------------------------ unsupervised

def build_autoencoder(n_features, seed=0):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    return keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(48, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(8,  activation="relu", name="bottleneck"),
        layers.Dense(24, activation="relu"),
        layers.Dense(48, activation="relu"),
        layers.Dense(n_features, activation="linear"),
    ], name="canguard_autoencoder")


def recon_error(ae, Xs, batch=1024):
    pred = ae.predict(Xs, batch_size=batch, verbose=0)
    return np.mean((pred - Xs) ** 2, axis=1)


def make_isolation_forest(seed=0):
    return IsolationForest(n_estimators=300, contamination=0.005,
                           random_state=seed, n_jobs=-1)


# -------------------------------------------------------------- supervised

def make_random_forest(seed=0):
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


def build_mlp(n_features, n_out=1, seed=0):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)
    out_act = "sigmoid" if n_out == 1 else "softmax"
    loss = "binary_crossentropy" if n_out == 1 else "sparse_categorical_crossentropy"
    m = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(n_out, activation=out_act),
    ], name="canguard_mlp" if n_out == 1 else "canguard_classifier")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    return m
