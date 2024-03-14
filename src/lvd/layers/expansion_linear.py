from flax import linen as nn

class ExpansionLinear(nn.Module):
    hidden_dim: int
    expansion: int

    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, training: bool = True):
        x = nn.Dense(self.expansion * self.hidden_dim)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
            
        x = nn.gelu(x)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        
        return x