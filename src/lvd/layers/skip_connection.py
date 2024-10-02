from typing import Type
from jax import Array

import flax.linen as nn

from lvd.config.network_config import SkipConnectionType


class SkipLayer(nn.Module):
     hidden_dim: int

     @nn.compact
     def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
          raise NotImplementedError

class Simple(SkipLayer):
     @nn.compact
     def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
          return inputs + state
        

class Identity(SkipLayer):
     @nn.compact
     def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
          return inputs

class OutputGate(SkipLayer):    
    @nn.compact
    def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
            gate = nn.sigmoid(nn.Dense(self.hidden_dim)(state))
            return state + gate * inputs
    

class Highway(SkipLayer):
    @nn.compact
    def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
            gate = nn.sigmoid(nn.Dense(self.hidden_dim)(state))
            return gate * state + (1 - gate) * inputs
    

class GRUSkip(SkipLayer):
    @nn.compact
    def __call__(self, inputs: Array, state: Array, *, training: bool = False) -> Array:
        return nn.GRUCell(self.hidden_dim)(state, inputs)[0]
    
    
def create_skip_connection(name: str) -> Type[SkipLayer]:
    if name == SkipConnectionType.Simple:
        return Simple
    elif name == SkipConnectionType.Identity:
        return Identity
    elif name == SkipConnectionType.Output:
        return OutputGate
    elif name == SkipConnectionType.Highway:
         return Highway
    elif name == SkipConnectionType.GRU:
         return GRUSkip
    else:
         raise ValueError(f"Unkown skip connection type: {name}")