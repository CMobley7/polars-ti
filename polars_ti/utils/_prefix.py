"""Preserve row alignment while initializing recurrences after missing prefixes."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ArrayKernel = Callable[[tuple[FloatArray, ...]], tuple[FloatArray, ...]]


def first_finite_index(arrays: Sequence[FloatArray]) -> int:
    """Return the first row where all inputs are finite, or their common length.

    Raises:
        ValueError: If inputs are absent, unequal in length, or not one-dimensional.
    """
    if not arrays:
        raise ValueError("at least one input array is required")
    if arrays[0].ndim != 1:
        raise ValueError("inputs must be one-dimensional arrays of equal length")
    size = len(arrays[0])
    valid = np.ones(size, dtype=np.bool_)
    for array in arrays:
        if array.ndim != 1 or len(array) != size:
            raise ValueError("inputs must be one-dimensional arrays of equal length")
        valid &= np.isfinite(array)
    positions = np.flatnonzero(valid)
    return int(positions[0]) if len(positions) else size


def run_after_prefix(
    arrays: tuple[FloatArray, ...],
    compute: ArrayKernel,
    outputs: int = 1,
    valid_inputs: int | None = None,
) -> tuple[FloatArray, ...]:
    """Run a recurrence on its finite suffix and restore leading missing rows.

    Only the leading run is removed; interior missing values retain the kernel's
    existing semantics. Derived inputs may have their own warmup, so valid_inputs
    can restrict prefix detection to the primary inputs at the front of arrays.
    Offsets must be applied after this function restores the original row count.

    Raises:
        ValueError: If input/output counts or kernel result shapes are inconsistent.
    """
    if outputs < 1 or (valid_inputs is not None and not 1 <= valid_inputs <= len(arrays)):
        raise ValueError("positive output count and a valid primary-input count are required")
    if valid_inputs is None:
        start = first_finite_index(arrays)
    else:
        first_finite_index(arrays)
        start = first_finite_index(arrays[:valid_inputs])
    size = len(arrays[0])
    if start == size:
        return tuple(np.full(size, np.nan) for _ in range(outputs))
    result = compute(tuple(array[start:] for array in arrays))
    if len(result) != outputs or any(array.ndim != 1 or len(array) != size - start for array in result):
        raise ValueError("kernel outputs must preserve the input row count")
    if start == 0:
        return result
    return tuple(np.concatenate((np.full(start, np.nan), array)) for array in result)
