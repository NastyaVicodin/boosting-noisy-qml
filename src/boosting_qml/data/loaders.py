"""Reusable JAX/Eqx data loading helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, NamedTuple, Protocol, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat
from equinox import Module, filter_jit, filter_vmap
from equinox.nn import Identity
from jaxtyping import Array, Bool, Int, PRNGKeyArray

T = TypeVar("T")
_I = TypeVar("_I")
_O = TypeVar("_O")


class TransT(Protocol[_I, _O]):
    """Transform protocol used by :class:`DataLoader`."""

    def __call__(self, x: _I, *, key: PRNGKeyArray | None) -> _O:
        ...


class DState(NamedTuple, Generic[T]):
    """Data loader state."""

    xs: T
    pad: Bool[Array, " b"]
    length: Int[Array, ""]
    epoch: Int[Array, ""] = jnp.asarray(0)
    step: Int[Array, ""] = jnp.asarray(0)
    key: PRNGKeyArray = jax.random.key(42)
    name: str | None = None


_IDX_FN = Callable[[Int[Array, ""]], DState[Int[Array, " b d"]]]


def _np_sort(x: jax.Array, axis: int | None = None):
    return np.argsort(np.asarray(x), axis=axis)


@eqx.filter_jit
def fallback_argsort(x: jax.Array, axis: int | None = None) -> jax.Array:
    """Fallback to a CPU-safe argsort callback when needed."""

    if jax.devices()[0].platform == "cpu":
        return jax.pure_callback(
            _np_sort,
            jax.ShapeDtypeStruct(x.shape, jnp.int32),
            x,
            axis,
        )
    return x.argsort(axis=axis)


def batch_index(
    length: int,
    batch_size: int,
    drop_last: bool = False,
    shuffle: bool = False,
    *,
    key: PRNGKeyArray | None = None,
) -> _IDX_FN:
    """Batchify integer indices."""

    r = length % batch_size
    new_length = (length - r) if drop_last else length
    pad = (batch_size - (new_length % batch_size)) % batch_size
    add_one = (not drop_last) and (r != 0)
    dlength = (length // batch_size) + (1 if add_one else 0)
    depoch = jnp.asarray(0)
    idxes_base = jnp.arange(length)

    @eqx.filter_jit
    def _index(epoch: Int[Array, ""] = depoch) -> DState[Int[Array, " b d"]]:
        base_key = key if key is not None else jax.random.PRNGKey(0)
        new_key = jax.random.fold_in(base_key, epoch)

        rand_order = jnp.take_along_axis(
            idxes_base,
            fallback_argsort(jax.random.uniform(new_key, shape=(length,))),
            axis=0,
        )
        idxes = jnp.where(shuffle, rand_order, idxes_base)

        pad_array = jnp.full((pad,), -1, dtype=idxes.dtype)
        idxes = jnp.concatenate([idxes, pad_array], axis=0)
        slice_size = new_length + pad
        idxes = idxes[:slice_size].reshape(-1, batch_size)

        pad_mask = jnp.where(idxes == -1, 1, 0).astype(bool)
        batch_keys = jax.random.split(new_key, idxes.shape[0])

        return DState(
            xs=idxes,
            pad=pad_mask,
            length=dlength,
            epoch=epoch,
            key=batch_keys,
        )

    return _index


class DataLoader(Module, Generic[T]):
    """Simple deterministic data loader."""

    gen: _IDX_FN
    embed: TransT[Int[Array, " d"], T]
    transform: TransT[T, T]
    _default_epoch: Int[Array, ""]
    _length: int
    _data_length: int
    _batch_size: int

    def __init__(
        self,
        length: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        key: PRNGKeyArray | None = None,
        *,
        embed: TransT[Int[Array, " d"], T] | None = None,
        transform: TransT[T, T] | None = None,
    ) -> None:
        self.gen = eqx.filter_jit(batch_index(length, batch_size, drop_last, shuffle, key=key))
        self.embed = (
            embed if embed is not None else cast(TransT[Int[Array, " d"], T], Identity())
        )
        self.transform = (
            transform if transform is not None else cast(TransT[T, T], Identity())
        )
        self._default_epoch = jnp.asarray(0)
        self._length = self.gen(self._default_epoch).length
        self._data_length = length
        self._batch_size = batch_size

    def __call__(self, epoch: Int[Array, ""] | None = None) -> Iterator[DState[T]]:
        # Keep epoch as a dynamic JAX scalar. Passing a Python int into the
        # jitted index function makes Equinox treat it as static, which can
        # trigger one compilation/cache entry per epoch in long EM runs.
        epoch_arg = self._default_epoch if epoch is None else jnp.asarray(epoch, dtype=self._default_epoch.dtype)
        data = self.gen(epoch_arg)

        for step, d, p, k in zip(jnp.arange(len(self)) + 1, data.xs, data.pad, data.key):
            yield DState(
                self.transform(self.embed(d, key=k), key=k),
                p,
                data.length,
                data.epoch,
                step,
                k,
            )

    @filter_jit
    def _ex(self, x: Array) -> Array:
        if x.ndim > 1:
            return x.reshape(-1, *x.shape[2:])
        return repeat(x, "e ... -> (e l) ...", l=self._length)

    def epoch_iter(
        self, epoch_start: int, epoch_end: int | None = None
    ) -> Iterator[DState[T]]:
        gen = filter_jit(filter_vmap(self.gen))
        if epoch_end is None:
            epoch_end = epoch_start + 1
            epoch_start = 1
        data = gen(jnp.arange(epoch_start, epoch_end))

        data = jax.tree.map(self._ex, data)
        for step, x, pad, length, epoch, key in zip(
            jnp.arange(data.xs.shape[0]) + 1,
            data.xs,
            data.pad,
            data.length,
            data.epoch,
            data.key,
        ):
            yield DState(
                self.transform(self.embed(x, key=key), key=key),
                pad,
                length,
                epoch,
                step,
                key,
            )

    def viter(
        self, epochs: int, chunks: int = 100, *, epoch_bias: int = 1
    ) -> Iterator[DState[T]]:
        for chunk in range(epoch_bias, epochs + epoch_bias, chunks):
            yield from self.epoch_iter(chunk, min(chunk + chunks, epochs + epoch_bias))

    def __len__(self) -> int:
        return self._length


__all__ = ["DState", "DataLoader", "batch_index"]
