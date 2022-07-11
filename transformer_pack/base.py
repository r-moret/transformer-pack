import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from sklearn.base import TransformerMixin, BaseEstimator
from functools import partial


class CategoricalEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, *, embedding_dim="auto", objective=None, random_state=None):
        self.embedding_dim = embedding_dim
        self.objective = objective
        self.random_state = random_state

        tf.random.set_seed(self.random_state)

    def fit(self, X, y=None):
        X = X.copy()

        n_features = X.shape[1]
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not np.issubdtype(y.dtype, np.integer) and not np.issubdtype(
            y.dtype, np.floating
        ):
            try:
                y = y.astype(float)
            except:
                raise ValueError("y must be of integer or float type.")

        self.embeddings_ = []
        for i_col in range(n_features):
            feature = X[:, i_col].astype(str).copy()

            model = self._embedding_model(feature)
            model.fit(feature, y, verbose=0)

            feature_embeddings = {
                key: list(embedding)
                for key, embedding in zip(
                    ["[UNK]"] + list(np.unique(feature)),
                    model.layers[2].get_weights()[0],
                )
            }

            self.embeddings_.append(feature_embeddings)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        Xt = pd.DataFrame() if isinstance(X, pd.DataFrame) else []

        n_features = X.shape[1]
        for i_col in range(n_features):
            embedder = partial(map, lambda x: self.embeddings_[i_col][x])

            if isinstance(X, pd.DataFrame):
                feature = X.iloc[:, i_col].astype(str).copy()
                embedded_feature = embedder(feature)
                Xt[X.iloc[:, i_col].name] = list(embedded_feature)
            else:
                feature = X[:, i_col].astype(str).copy()
                embedded_feature = embedder(feature)
                Xt.append(list(embedded_feature))

        if not isinstance(X, pd.DataFrame):
            Xt = np.array(Xt, dtype='object').T

        return Xt

    def _embedding_model(self, feature):
        vocab = np.unique(feature)
        n_uniques = len(vocab)

        if self.embedding_dim == "auto":
            feature_embedding_dim = int(np.sqrt(n_uniques))
        else:
            feature_embedding_dim = int(self.embedding_dim)

        if self.objective == "binary-class":
            activation_fun = "sigmoid"
            loss_fun = "binary_crossentropy"
        elif self.objective == "regression":
            activation_fun = "linear"
            loss_fun = "mse"
        elif self.objective == "multi-class":
            raise NotImplementedError("This feature is still on development.")
        else:
            raise ValueError(
                "Not supported objective. Please specify one of 'binary-class', "
                "'multi-class' or 'regression', depending on your problem."
            )

        input_layer = tf.keras.layers.Input(shape=(1,), dtype="string")
        embedding_nn = tf.keras.layers.StringLookup(vocabulary=vocab)(input_layer)
        embedding_nn = tf.keras.layers.Embedding(
            output_dim=feature_embedding_dim, input_dim=n_uniques + 1, input_length=1
        )(embedding_nn)
        embedding_nn = tf.keras.layers.Flatten()(embedding_nn)
        embedding_nn = tf.keras.layers.Dense(units=128, activation="relu")(embedding_nn)
        embedding_nn = tf.keras.layers.Dense(units=1, activation=activation_fun)(
            embedding_nn
        )

        model = tf.keras.Model(input_layer, embedding_nn)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(), loss=loss_fun,
        )

        return model
