from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Any
import numpy as np


class BaseChannelSelector:
    """
    Base class for channel selection / dropping transforms.

    Subclasses implement `_choose_indices(n_channels, **kwargs)` to return the
    list/array of channel indices to KEEP when the transform is applied.

    Calling the instance like a function applies the transform to *all* inputs
    passed via *args, each of which must share the same leading channel axis.

    Convention:
        - If the transform is applied (based on `p`), returns a list where each
          element is the input sliced by the chosen indices along the first axis.
        - If the transform is NOT applied, returns the original `args` unchanged.

    Notes:
        - This class uses a NumPy Generator for randomness. Provide a fixed
          `rng` for reproducible sampling.
    """

    def __init__(self, p: float = 1.0, rng: Optional[np.random.Generator] = None):
        """
        Args:
            p: Probability in [0, 1] of applying the transform on call.
            rng: Optional NumPy random Generator for reproducibility.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- hooks for subclasses -------------------------------------------------

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        """
        Return the indices to KEEP (np.int64 array of shape (k, )).

        Subclasses MUST implement this.
        """
        raise NotImplementedError

    # ---- core call ------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs) -> List[Any] | Tuple[Any, ...]:
        """
        Apply the selection to all positional inputs.

        Returns:
            - If applied: list of inputs sliced on the first axis by the chosen indices.
            - If skipped: the original `args` (tuple) unchanged.

        Raises:
            AssertionError: if inputs have mismatched number of channels on axis 0.
        """
        # Decide whether to apply
        if (self.p < 1.0) and (self.rng.uniform() >= self.p):
            return args  # unchanged

        if len(args) == 0:
            return []

        # Validate all inputs share the same number of channels on axis 0
        lengths = {len(arg) for arg in args}
        assert len(lengths) == 1, "All inputs must have the same number of channels"
        n_channels = len(args[0])

        # Get indices to KEEP from subclass
        keep = np.asarray(self._choose_indices(n_channels, **kwargs), dtype=np.int64)
        if keep.ndim != 1:
            raise ValueError("Indices must be a 1D array")
        if keep.size == 0:
            # Nothing selected: return empty views of the inputs along axis 0
            return [arg[keep] for arg in args]

        # Slice all inputs on the first axis
        return [arg[keep] for arg in args]


class DropChannelsFraction(BaseChannelSelector):
    """
    Randomly keep a fraction of channels (uniformly sampled within a range).

    Behavior:
        - Sample fraction f ~ Uniform([f_min, f_max]).
        - Keep ceil(f * N) distinct channels chosen uniformly without replacement.

    Args:
        p: Probability of applying the transform.
        fraction_range: (f_min, f_max) with 0 <= f_min <= f_max <= 1.
        rng: Optional NumPy RNG for reproducibility.

    Example:
        >>> t = DropChannelsFraction(p=1.0, fraction_range=(0.5, 0.75))
        >>> x, y = np.arange(10)[:,None], np.arange(10)[:,None]
        >>> x2, y2 = t(x, y)  # both have the same subset of rows
    """

    def __init__(
        self,
        p: float = 0.5,
        fraction_range: Tuple[float, float] = (0.5, 0.5),
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(p=p, rng=rng)
        fmin, fmax = fraction_range
        if not (0.0 <= fmin <= fmax <= 1.0):
            raise ValueError("fraction_range must satisfy 0 <= fmin <= fmax <= 1")
        self.fraction_range = (float(fmin), float(fmax))

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        f = self.rng.uniform(self.fraction_range[0], self.fraction_range[1])
        k = int(np.ceil(n_channels * f))
        k = max(0, min(k, n_channels))
        return self.rng.choice(n_channels, size=k, replace=False)


class DropChannelsFixedNumber(BaseChannelSelector):
    """
    Randomly keep a fixed number of channels.

    Args:
        p: Probability of applying the transform.
        num_keep: Number of channels to keep (will be clamped to [0, N]).
        rng: Optional NumPy RNG for reproducibility.

    Example:
        >>> t = DropChannelsFixedNumber(p=1.0, num_keep=3)
        >>> (x2,) = t(np.arange(8))
        >>> len(x2) == 3
        True
    """

    def __init__(
        self,
        p: float = 0.5,
        num_keep: int = 1,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(p=p, rng=rng)
        if num_keep < 0:
            raise ValueError("num_keep must be >= 0")
        self.num_keep = int(num_keep)

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        k = min(self.num_keep, n_channels)
        return self.rng.choice(n_channels, size=k, replace=False)


class DropChannelsFixedNumberRange(BaseChannelSelector):
    """
    Randomly keep a number of channels sampled uniformly from [min, max] (inclusive).

    Args:
        p: Probability of applying the transform.
        num_keep_min: Minimum number of channels to keep.
        num_keep_max: Maximum number of channels to keep (must be >= min).
        rng: Optional NumPy RNG for reproducibility.

    Example:
        >>> t = DropChannelsFixedNumberRange(p=1.0, num_keep_min=2, num_keep_max=5)
        >>> x2, = t(np.arange(10))
        >>> 2 <= len(x2) <= 5
        True
    """

    def __init__(
        self,
        p: float = 0.5,
        num_keep_min: int = 1,
        num_keep_max: int = 1,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(p=p, rng=rng)
        if num_keep_min < 0 or num_keep_max < 0 or num_keep_min > num_keep_max:
            raise ValueError("Require 0 <= num_keep_min <= num_keep_max")
        self.num_keep_min = int(num_keep_min)
        self.num_keep_max = int(num_keep_max)

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        if n_channels == 0:
            return np.array([], dtype=np.int64)
        k = self.rng.integers(self.num_keep_min, self.num_keep_max + 1)
        k = int(np.clip(k, 0, n_channels))
        return self.rng.choice(n_channels, size=k, replace=False)


class DropChannelsNuclearKnown(BaseChannelSelector):
    """
    Keep a fixed 'nuclear' channel (by index) plus a random subset of the remaining channels.

    Behavior:
        - Always include `fixed_index` in the kept set.
        - If `num_choose == -1`, include *all* remaining channels (i.e., keep everything).
        - Else, include `num_choose` additional channels sampled uniformly without replacement
          from the remaining indices. If fewer channels exist, keep all remaining.

    Args:
        num_choose: Number of non-nuclear channels to keep, or -1 to keep all.
        p: Probability of applying the transform.
        rng: Optional NumPy RNG for reproducibility.

    Call signature:
        __call__(*args, fixed_index: int)

    Example:
        >>> t = DropChannelsNuclearKnown(num_choose=2, p=1.0)
        >>> x, y = np.arange(6), np.arange(6)*10
        >>> x2, y2 = t(x, y, fixed_index=0)  # will always contain index 0
    """

    def __init__(
        self,
        num_choose: int = 2,
        p: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(p=p, rng=rng)
        if num_choose < -1:
            raise ValueError("num_choose must be >= -1")
        self.num_choose = int(num_choose)

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        if "fixed_index" not in kwargs or kwargs["fixed_index"] is None:
            raise ValueError("`fixed_index` must be provided to DropChannelsNuclearKnown")
        fixed_index = int(kwargs["fixed_index"])
        if not (0 <= fixed_index < n_channels):
            raise IndexError(f"fixed_index {fixed_index} out of range for {n_channels} channels")

        remaining = np.setdiff1d(np.arange(n_channels, dtype=np.int64), np.array([fixed_index]))
        if self.num_choose == -1:
            keep = np.concatenate(([fixed_index], remaining))
        else:
            k = min(self.num_choose, remaining.size)
            chosen = self.rng.choice(remaining, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
            keep = np.concatenate(([fixed_index], chosen))
        return keep


class HierarchicalChannelSampling(BaseChannelSelector):
    """
    Keep a random number of channels sampled uniformly from [min_channels, N].

    (This mirrors the original `HierchicalChannelSampling` behavior but with consistent
    naming and base-class integration.)

    Args:
        min_channels: Minimum number of channels to keep (inclusive).
        p: Probability of applying the transform (default 1.0 for always-apply).
        rng: Optional NumPy RNG for reproducibility.

    Example:
        >>> t = HierarchicalChannelSampling(min_channels=3, p=1.0)
        >>> (x2,) = t(np.arange(10))
        >>> len(x2) >= 3
        True
    """

    def __init__(
        self,
        min_channels: int = 1,
        p: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(p=p, rng=rng)
        if min_channels < 0:
            raise ValueError("min_channels must be >= 0")
        self.min_channels = int(min_channels)

    def _choose_indices(self, n_channels: int, **kwargs) -> np.ndarray:
        if n_channels == 0:
            return np.array([], dtype=np.int64)
        low = min(self.min_channels, n_channels)
        k = int(self.rng.integers(low, n_channels + 1))  # inclusive upper bound
        return self.rng.choice(n_channels, size=k, replace=False)
