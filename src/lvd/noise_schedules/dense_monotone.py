from jax import Array
from jax import numpy as jnp
from jax import lax

from flax import linen as nn

class DenseMonotone(nn.Dense):

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        # Monotone network has all positive weights
        kernel = jnp.square(kernel)

        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = nn.linear.promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        return y