from jax import Array

from flax import linen as nn

from lvd.layers.skip_connection import create_skip_connection

class AttentionBlock(nn.Module):
    hidden_dim: int
    num_heads: int

    dropout: float = 0.0
    skip_connection_type: str = "gru"

    def setup(self):
        self.norm = nn.LayerNorm()

        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout
        )

        self.skip_connection = create_skip_connection(self.skip_connection_type)(self.hidden_dim)
        
    def __call__(
        self, 
        embeddings: Array,
        mask: Array,
        *,
        training: bool = False,
    ) -> Array:
        """
        Parameters
        ----------
        embeddings: (B, T, D)
        mask: (B, T)
        is_training: bool, keyword-only

        Returns
        -------
        embeddings: (B, T, D)
        """  
        
        square_mask = mask[:, None, None, :] & mask[:, None, :, None]

        hidden = self.norm(embeddings)
        hidden = self.attention(hidden, hidden, mask=square_mask, deterministic=not training)
        hidden = self.skip_connection(hidden, embeddings)
        
        return hidden