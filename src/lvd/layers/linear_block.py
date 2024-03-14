from flax import linen as nn

from lvd.utils import masked_fill
from lvd.layers.expansion_linear import ExpansionLinear
from lvd.layers.skip_connection import create_skip_connection

class LinearBlock(nn.Module):
    hidden_dim: int
    expansion: int = 2

    dropout: float = 0.0
    skip_connection_type: str = "gru"

    def setup(self):
        self.norm = nn.LayerNorm()
        self.expansion_linear = ExpansionLinear(self.hidden_dim, self.expansion, self.dropout)
        self.skip_connection = create_skip_connection(self.skip_connection_type)(self.hidden_dim)

    def __call__(self, x, *, training: bool = True):
        y = self.norm(x)
        y = self.expansion_linear(y, training = training)
        y = self.skip_connection(y, x)

        return y