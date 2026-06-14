from dataclasses import dataclass
from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from jax import lax

from spargel_llm.layers.jax import (
    DotProductAttention,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    PositionEncodingStrategy,
    StaticPositionEncoding,
    all_you_need_init,
)


@dataclass
class LLMConfig:
    vocab_size: int
    embed_dim: int
    qk_dim: int
    hidden_dim: int
    block_size: int
    layer_count: int

    position_encoding: PositionEncodingStrategy = PositionEncodingStrategy.ALL_YOU_NEED


class LLM(nnx.Module):
    config: LLMConfig
    attentions: list[DotProductAttention]
    feed_forwards: list[FeedForward]

    def __init__(
        self,
        config: LLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.token_embed = Embedding(config.vocab_size, config.embed_dim, rngs=rngs)

        match config.position_encoding:
            case PositionEncodingStrategy.NONE:
                self.pos_embed = None
            case PositionEncodingStrategy.LEARNED:
                self.pos_embed = Embedding(
                    config.block_size, config.embed_dim, rngs=rngs
                )
            case PositionEncodingStrategy.ALL_YOU_NEED:
                self.pos_embed = StaticPositionEncoding(
                    config.block_size, config.embed_dim, all_you_need_init
                )
            case _:
                raise Exception("unimplmented")

        self.embed_normalization = LayerNorm(config.embed_dim, rngs=rngs)

        self.attentions = []
        self.feed_forwards = []

        for _ in range(config.layer_count):
            self.attentions.append(
                DotProductAttention(
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.qk_dim,
                    rngs=rngs,
                )
            )
            self.feed_forwards.append(
                FeedForward(
                    config.embed_dim, config.embed_dim, config.hidden_dim, rngs=rngs
                )
            )
        self.lm_head = Linear(config.embed_dim, config.vocab_size, rngs=rngs)

    def __call__(self, input: jax.Array, padding_mask: jax.Array):
        """
        Args:
            input: shape (..., seq_len). an array of token ids
            padding_mask (jax.Array): (..., seq_len)
        """
        seq_len = input.shape[-1]
        assert seq_len == self.config.block_size
        assert seq_len == padding_mask.shape[-1]

        mask = padding_mask[:, jnp.newaxis] & padding_mask[..., jnp.newaxis]

        causal_mask = jnp.tri(seq_len, dtype=jnp.bool)
        mask = mask & causal_mask

        x = self.token_embed(input)

        if self.pos_embed is not None:
            x += self.pos_embed(jnp.arange(seq_len))

        for i in range(self.config.layer_count):
            x_norm = self.embed_normalization(x)
            x += self.attentions[i](x_norm, x_norm, x_norm, mask=mask)
            x_norm = self.embed_normalization(x)
            x += self.feed_forwards[i](x_norm)

        x_norm = self.embed_normalization(x)
        return self.lm_head(x_norm)


def loss_fn(model: LLM, x: jax.Array, y: jax.Array):
    """
    Args:
        x: (batch_size, seq_len) dtype=int
        y: (batch_size, seq_len) dtype=int
    """
    assert x.shape == y.shape
    shape = x.shape
    mask = jnp.full(shape, True, dtype=jnp.bool)
    logits = model(x, padding_mask=mask)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels=y))


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int
    seq_len: int


@partial(jax.jit, static_argnames=["config"])
def get_batch(key, data, config: BatchConfig):
    """
    Args:
        key: the random key to be consumed
    """
    # the second argument is `ix` (the only one that is batched)
    dynamic_slice_vmap = jax.vmap(lax.dynamic_slice, in_axes=(None, 0, None))
    ix = jax.random.randint(
        key,
        shape=(config.batch_size, 1),
        minval=0,
        maxval=len(data) - config.seq_len,
    )
    x = dynamic_slice_vmap(data, ix, (config.seq_len,))
    y = dynamic_slice_vmap(data, ix + 1, (config.seq_len,))
    return x, y


@partial(nnx.jit, static_argnames=["config"])
def train_step(model, key, optimizer, metrics, data, config: BatchConfig):
    key, subkey = jax.random.split(key)
    batch = get_batch(subkey, data, config)
    loss, grads = nnx.value_and_grad(loss_fn)(model, *batch)
    metrics.update(values=loss)
    optimizer.update(model, grads)
    return loss, key


class TrainLoop:
    def __init__(
        self, model: LLM, learning_rate: float, batch_size: int, seq_len: int, data
    ):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
        self.data = data
        self.config = BatchConfig(batch_size, seq_len)
        self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average())

    def train_step(self, key):
        return train_step(
            self.model, key, self.optimizer, self.metrics, self.data, self.config
        )
