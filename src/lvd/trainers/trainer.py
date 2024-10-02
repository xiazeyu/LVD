from typing import NamedTuple, Callable, Tuple, Dict, Optional, Any

from jax import Array
from jax.random import PRNGKey

from flax.linen import Module
from flax.training.train_state import TrainState

from lvd.dataset import Dataset, Batch


InitializeType = Callable[[PRNGKey, Dataset, Batch], TrainState]
UpdateType = Callable[[TrainState, Batch], Tuple[TrainState, Dict[str, Array]]]
GenerateType = Callable[[
    TrainState, 
    Array, 
    Array, 
    Array, 
    int, 
    Optional[PRNGKey]
], Any]

ReconstructType = Callable[[TrainState, Array, Optional[PRNGKey]], Any]


class Trainer(NamedTuple):
    model: Module
    initialize: InitializeType
    update: UpdateType
    generate: GenerateType
    reconstruct: ReconstructType