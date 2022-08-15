import math
import multiprocessing
from functools import partial
from itertools import chain, islice
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn
from tqdm import trange

from .sklearn_wrappers import DFVectorizer, Dropper, Selector
from .types import Feature, PredictionFeature, Target

# Batch significantly more than 2,000,000 leads to memory issues but works
# well around 2,000,000
PREDICTION_BATCH_SIZE = 2000000


class EmbeddedRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = [100],
        batch_size: int = 200,
        solver: str = "sgd",
        learning_rate_init: float = 0.01,
        max_iter: int = 10,
        activation: str = "relu",
        dropout: float = 0.5,
        batch_normalization: bool = True,
        weight_decay: float = 0.0001,
        gpu_count: int = torch.cuda.device_count(),
        loss: str = "l2",
        embedding_rule: int = 0,
        cpu_workers: int = 0,
        no_embedding: List[str] = [],
        no_base: List[str] = [],
        random_state=None,  # needed for AverageEstimator
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.activation = activation
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.loss = loss
        self.embedding_rule = embedding_rule
        self.cpu_workers = cpu_workers
        self.no_embedding = no_embedding
        self.no_base = no_base
        self.random_state = random_state
        self.gpu_count = gpu_count

    def fit(self, X: Feature, y: Target, **fit_params) -> "EmbeddedRegression":
        self.identify_columns_to_embed(X)
        self.create_feature_pipelines()
        self.create_network(X, y)
        self.fit_(X, y, sample_weight=fit_params.get("sample_weight"))
        return self

    def identify_columns_to_embed(self, X: Feature) -> None:
        self.columns_to_embed = [
            column for column in string_columns(X) if column not in self.no_embedding
        ]
        self.embedding_input_dims = [
            X[column].nunique() + 1 for column in self.columns_to_embed
        ]

    def create_feature_pipelines(self) -> None:
        self.base_feature_pipeline = make_pipeline(
            Dropper(self.columns_to_embed + self.no_base),
            DFVectorizer(sparse=False),
            SimpleImputer(),
            StandardScaler(),
            FunctionTransformer(torch.tensor, validate=True),
        )
        self.embedding_feature_pipelines = [
            make_pipeline(
                Selector([column]),
                DFVectorizer(sparse=True),
                FunctionTransformer(
                    self.one_hot_to_ordinal, validate=True, accept_sparse=True
                ),
            )
            for column in self.columns_to_embed
        ]
        self.pipelines = [self.base_feature_pipeline, *self.embedding_feature_pipelines]

    def create_network(self, X: Feature, y: Target) -> None:
        self.net_ = EmbeddingMLPRegressor(
            self.base_input_dim(X),
            self.embedding_input_dims,
            self.hidden_layer_sizes,
            self.output_dimension(y),
            self.activation,
            self.dropout,
            self.batch_normalization,
            self.embedding_rule,
        )
        self.net_.apply(init_weights)
        self.nets = [self.net_]

    def fit_(
        self, X: Feature, y: Target, sample_weight: Optional[np.ndarray] = None
    ) -> None:
        self.train_model()
        optimizer = self.optimizer
        sample_weight = (
            sample_weight if sample_weight is not None else np.ones(y.shape[0])
        )
        data_loader = self.data_loader(X, y, sample_weight)
        for _ in trange(self.max_iter):
            for *X_batch, y_batch, sample_weight_batch in data_loader:
                optimizer.zero_grad()
                output = self.forward(*X_batch)
                losses = self.criterion(output, y_batch.float())
                loss = torch.mean(losses * sample_weight_batch.float())
                loss.backward()
                optimizer.step()

    def train_model(self) -> None:
        for net in self.nets:
            net.train()

    def forward(self, *args):
        return self.net_(*args)

    def data_loader(self, X: Feature, y: Target, sample_weight: np.ndarray):
        dataset = data_utils.TensorDataset(
            *self.feature_tensors(X, train=True),
            self.y_tensor(y),
            self.sample_weight_tensor(sample_weight),
        )
        return data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def feature_tensors(self, X: Feature, train: bool):
        transform = "fit_transform" if train else "transform"
        return [getattr(pipeline, transform)(X) for pipeline in self.pipelines]

    def predict(self, X: PredictionFeature):
        return np.concatenate(
            [
                self._predict(X_batch)
                for X_batch in self.chunker(X, PREDICTION_BATCH_SIZE)
            ]
        )

    def eval_model(self):
        for net in self.nets:
            net.eval()

    def _predict(self, X: PredictionFeature):
        self.eval_model()
        prediction = (
            self.forward(*self.feature_tensors(X, train=False))
            .type(torch.DoubleTensor)
            .detach()
            .numpy()
        )
        if prediction.shape[1] == 1:
            return prediction.ravel()
        return prediction

    @staticmethod
    def chunker(seq, size):
        if isinstance(seq, dict):
            return [seq]
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    @staticmethod
    def one_hot_to_ordinal(X: csr_matrix) -> torch.LongTensor:
        return torch.LongTensor(X.dot(np.arange(1, X.shape[1] + 1)))

    @property
    def criterion(self):
        loss_functions = {
            "l1": nn.L1Loss(reduction="none"),
            "huber": nn.SmoothL1Loss(reduction="none"),
            "l2": nn.MSELoss(reduction="none"),
            "ordinal": partial(ordinal_loss, reduction="none"),
        }
        try:
            return loss_functions[self.loss]
        except KeyError:
            return loss_functions["l2"]

    @property
    def optimizer(self):
        optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        try:
            optimizer = optimizers[self.solver]
        except KeyError:
            optimizer = optimizers["adam"]
        return optimizer(
            self.net_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.weight_decay,
        )

    def base_input_dim(self, X):
        self.base_feature_pipeline.fit(X)
        return self.base_feature_pipeline.transform(X.iloc[[0]]).shape[1]

    @staticmethod
    def output_dimension(y: Target) -> int:
        if y.ndim == 1:
            return 1
        if y.ndim == 2:
            return y.shape[1]
        raise ValueError("only 1 and 2d supported")

    @property
    def num_workers(self) -> int:
        if self.gpu_count > 0:
            return self.cpu_workers or multiprocessing.cpu_count()
        return 0

    @staticmethod
    def y_tensor(y: Target):
        y = y if isinstance(y, np.ndarray) else y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return torch.tensor(y)  # pylint: disable=not-callable

    @staticmethod
    def sample_weight_tensor(w: np.ndarray) -> torch.Tensor:
        return torch.tensor(w.reshape(-1, 1))  # pylint: disable=not-callable


class SequenceEmbeddedRegression(EmbeddedRegression):
    def __init__(
        self,
        timeseries_features: List[str],
        initialization_features: List[str],
        hidden_layer_sizes: Sequence[int] = [100],
        batch_size: int = 200,
        solver: str = "sgd",
        learning_rate_init: float = 0.01,
        max_iter: int = 10,
        activation: str = "relu",
        dropout: float = 0.5,
        batch_normalization: bool = True,
        weight_decay: float = 0.0001,
        gpu_count: int = torch.cuda.device_count(),
        loss: str = "l2",
        embedding_rule: int = 0,
        cpu_workers: int = 0,
        no_embedding: List[str] = [],
        random_state=None,
        timeseries_output_dim=20,
        stacked_layers=1,
    ):
        self.timeseries_features = timeseries_features
        self.initialization_features = initialization_features
        self.timeseries_output_dim = timeseries_output_dim
        self.stacked_layers = stacked_layers
        self.batch_first = True
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            dropout=dropout,
            batch_normalization=batch_normalization,
            weight_decay=weight_decay,
            gpu_count=gpu_count,
            loss=loss,
            embedding_rule=embedding_rule,
            cpu_workers=cpu_workers,
            no_embedding=no_embedding,
            no_base=self.timeseries_features + self.initialization_features,
            random_state=random_state,
        )

    def create_feature_pipelines(self) -> None:
        super().create_feature_pipelines()

        self.initialization_feature_pipeline = make_pipeline(
            Selector(self.initialization_features),
            DFVectorizer(sparse=False),
            SimpleImputer(),
            StandardScaler(),
            FunctionTransformer(torch.tensor, validate=True),
        )
        self.pipelines.append(self.initialization_feature_pipeline)

        sr = SequenceReshaper()
        # timeseries_feature_pipeline is not appended to self.pipelines
        # because the output needs additional processing. Hence,
        # timeseries_feature_pipeline is called separately.
        self.timeseries_feature_pipeline = make_pipeline(
            Selector(self.timeseries_features),
            FunctionTransformer(delta_array, validate=False),
            FunctionTransformer(sr.to_1d, validate=False),
            StandardScaler(),
            FunctionTransformer(sr.to_tensor, validate=False),
        )

    def create_network(self, X, y):
        super().create_network(X, y)
        self.rnn_ = SequenceRNN(
            self.initialization_input_dim(X),
            self.timeseries_input_dim,
            self.stacked_layers,
            self.timeseries_output_dim,
            self.batch_first,
        )
        self.rnn_.apply(init_weights)
        self.nets.append(self.rnn_)

    def forward(self, *args):
        # arguments ordering should match feature_tensors ordering
        base, *embeddings, init_input, timeseries_features, tensor_size = args
        return self.net_(
            torch.cat(
                [self.rnn_(timeseries_features, tensor_size, init_input), base.float()],
                dim=1,
            ),
            *embeddings,
        )

    def feature_tensors(self, X: Feature, train: bool):
        return super().feature_tensors(X, train) + list(
            self.timeseries_tensors(X, train)
        )

    def timeseries_tensors(self, X, train: bool):
        transform = "fit_transform" if train else "transform"
        samples = getattr(self.timeseries_feature_pipeline, transform)(X)
        sample_lengths = torch.tensor(  # pylint: disable=not-callable
            [len(sample) for sample in samples]
        )
        padded = nn.utils.rnn.pad_sequence(samples, batch_first=self.batch_first)
        return padded, sample_lengths

    def base_input_dim(self, X):
        return super().base_input_dim(X) + self.timeseries_output_dim

    @property
    def timeseries_input_dim(self):
        return len(self.timeseries_features)

    def initialization_input_dim(self, X):
        self.initialization_feature_pipeline.fit(X)
        return self.initialization_feature_pipeline.transform(X.iloc[[0]]).shape[1]


class EmbeddingMLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_input_dims: Sequence[int],
        hidden_layer_sizes: Sequence[int],
        output_dimension: int,
        activation: str = "relu",
        dropout: float = 0.5,
        batch_normalization: bool = False,
        embedding_rule: int = 0,
    ):
        super().__init__()
        self.base_input_dim = input_dim
        self.embedding_input_dims = embedding_input_dims
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_dimension = output_dimension
        self.dropout = dropout
        self.has_batch_normalization = batch_normalization
        self.embedding_rule = embedding_rule
        self._activation = activation

        self.create_embedding()
        self.create_sequential()

    def create_embedding(self):
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(input_dim, embedding_rule(input_dim, self.embedding_rule))
                for input_dim in self.embedding_input_dims
            ]
        )

    def create_sequential(self):
        layers = []

        for size_in, size_out in zip(
            [self.sequential_input_dim] + self.hidden_layer_sizes,
            self.hidden_layer_sizes,
        ):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(self.activation)
            if self.has_batch_normalization:
                layers.append(nn.BatchNorm1d(size_out))
        if self.dropout:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        feature, *embedding_features = args
        embedded_features = [
            embedding(embedding_feature)
            for embedding, embedding_feature in zip(self.embeddings, embedding_features)
        ]
        return self.net(torch.cat([feature.float()] + embedded_features, dim=1))

    @property
    def sequential_input_dim(self):
        return self.base_input_dim + sum(
            [
                embedding_rule(input_dim, self.embedding_rule)
                for input_dim in self.embedding_input_dims
            ]
        )

    @property
    def activation(self):
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        try:
            return activations[self._activation]
        except KeyError:
            return activations["relu"]


def embedding_rule(input_dim: int, i: int) -> int:
    if i == 0:
        return math.ceil(input_dim ** 0.25)
    if i == 1:
        return min(50, (input_dim + 1) // 2)
    return 10


def string_columns(X) -> List[str]:
    return [column for column in X.columns if isinstance(X[column].iloc[0], str)]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


class SequenceRNN(nn.Module):
    def __init__(
        self,
        initialization_input_dim: int,
        timeseries_input_dim: int = 1,
        stacked_layers: int = 1,
        timeseries_output_dim: int = 20,
        batch_first: bool = True,
    ):
        super().__init__()
        self.initialization_input_dim = initialization_input_dim
        self.timeseries_input_dim = timeseries_input_dim
        self.stacked_layers = stacked_layers
        self.timeseries_output_dim = timeseries_output_dim
        self.batch_first = batch_first
        self.create_rnn()

    def create_rnn(self):
        self.gru_initialization = nn.Sequential(
            nn.Linear(self.initialization_input_dim, self.timeseries_output_dim),
            nn.Sigmoid(),
        )
        self.rnn = nn.GRU(
            self.timeseries_input_dim,
            self.timeseries_output_dim,
            self.stacked_layers,
            batch_first=self.batch_first,
        )

    def forward(self, timeseries_features, tensor_size, init_input):
        packed_sequence, _ = self.rnn(
            nn.utils.rnn.pack_padded_sequence(
                timeseries_features.float(),
                tensor_size,
                enforce_sorted=False,
                batch_first=self.batch_first,
            ),
            self.gru_initialization(init_input.float())
            .unsqueeze(0)
            .repeat(self.stacked_layers, 1, 1),
        )

        output, output_lens = nn.utils.rnn.pad_packed_sequence(
            packed_sequence, batch_first=self.batch_first
        )

        return output[range(len(output_lens)), output_lens - 1, :]


class SequenceReshaper:
    """
    Each sample can have multiple sequence features
    where each sequence can have different lengths
    for different samples.
    input = [
            ([10, 10], [12, 12]),
            ([1, 1, 2], [2, 3, 4])
            ]

    Here the two input sequences are of size 2, 3 respectively.

    to_1d stacks same feature timeseries to apply standardization
    to_1d_output = [[10, 12],
                    [10, 12],
                    [ 1,  2],
                    [ 1,  3],
                    [ 2,  4]]

    to_tensor converts 1d_input to tensor
    to_tensor_output = [
                        tensor(
                            [[10, 12],
                            [10, 12]]
                        ),
                        tensor(
                            [[ 1, 2],
                            [ 1, 3],
                            [ 2, 4]]
                        )
                        ]
    """

    def __init__(self):
        self.sequence_lengths = []

    def to_1d(self, x):
        self.sequence_lengths = [len(i[0]) for i in x]
        return np.swapaxes(
            np.array(
                [
                    list(chain.from_iterable(i[qty_index] for i in x))
                    for qty_index in range(len(x[0]))
                ]
            ),
            0,
            1,
        )

    def to_tensor(self, x):
        x = iter(x)
        return [
            torch.tensor(list(islice(x, elem)))  # pylint: disable=not-callable
            for elem in self.sequence_lengths
        ]


def delta_array(X):
    if isinstance(X, list):
        X = pd.DataFrame(X)
    result = []
    for _, row in X.iterrows():
        row_result = [[column[-1] - i for i in column] for column in row]
        # If the list of sequence features is empty we need to insert 0.0
        # Current torch behaviour for torch.tensor([[]]) produces
        # single dimension tensor i.e. tensor([])
        if not row_result[0]:
            row_result = [[0.0]]
        result.append(row_result)
    return result


thresholds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


def hinge(z):
    return torch.max(torch.zeros_like(z), -z)


def ordinal_loss(y_pred, y_true, thresholds=thresholds, loss=hinge, reduction="none"):
    result = torch.zeros_like(y_pred)
    for threshold in thresholds:
        s = torch.sign(threshold - y_true)
        result += loss(s * (threshold - y_pred))
    if reduction == "none":
        return result
    return result.mean()
