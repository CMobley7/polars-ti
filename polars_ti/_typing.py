# -*- coding: utf-8 -*-
from decimal import Decimal
from functools import partial
from pathlib import Path
from sys import float_info as sflt
from typing import *

import numpy as np
from numpy import argmax, argmin
from numpy import floating as np_floating
from numpy import generic as np_generic
from numpy import integer as np_integer
from numpy import ndarray, recarray, void
from pandas import DataFrame, Series

nan = np.nan  # NumPy 2.0+ compatible

# Generic types
T = TypeVar("T")

# Scalars
Scalar = str | float | int | complex | bool | object
Number = int | float | complex
IntFloat = int | float
Int = int | None
Float = float | None

# Basic sequences
MaybeTuple = T | Tuple[T, ...]
MaybeList = T | List[T]
TupleList = List[T] | Tuple[T, ...]
MaybeTupleList = T | List[T] | Tuple[T, ...]
MaybeIterable = T | Iterable[T]
MaybeSequence = T | Sequence[T]
ListStr = List[str]

DictLike = dict | None
DictLikeSequence = MaybeSequence[DictLike]
Args = Tuple[Any, ...]
ArgsLike = Args | None
Kwargs = Dict[str, Any]
KwargsLike = Kwargs | None
KwargsLikeSequence = MaybeSequence[KwargsLike]
FileName = str | Path

DTypeLike = Any
PandasDTypeLike = Any
Shape = Tuple[int, ...]
RelaxedShape = int | Shape
Array = ndarray
Array1d = ndarray
Array2d = ndarray
Array3d = ndarray
Record = void
RecordArray = ndarray
RecArray = recarray
MaybeArray = T | Array
SeriesFrame = Series | DataFrame
MaybeSeries = T | Series
MaybeSeriesFrame = T | Series | DataFrame
AnyArray = Array | Series | DataFrame
AnyArray1d = Array1d | Series
AnyArray2d = Array2d | DataFrame

# =============================================================================
# Polars Type Aliases (for Polars-TI conversion)
# =============================================================================
import polars as pl

# Core Polars expression type - functions return this for lazy evaluation
PlExpr = pl.Expr

# Input type for indicator functions - accepts column name or expression
IntoExpr = pl.Expr | str

# Polars DataFrame types
PolarsFrame = pl.DataFrame | pl.LazyFrame
PlDataFrame = pl.DataFrame
PlLazyFrame = pl.LazyFrame

# Dual-mode support for migration period
DualFrame = DataFrame | pl.DataFrame | pl.LazyFrame
DualSeries = Series | pl.Series

# Polars-specific optional types
PlExprOpt = pl.Expr | None

