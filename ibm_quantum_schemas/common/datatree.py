# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DataTreeModel"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from typing_extensions import TypeAliasType

from ibm_quantum_schemas.aliases import Self
from ibm_quantum_schemas.common.tensor import CompressedTensorModel

DataTree: TypeAlias = (
    list["DataTree"] | dict[str, "DataTree"] | NDArray[float] | str | float | int | bool | None
)
"""User facing arbitrary nesting of lists and dicts with typed leaves."""

# TypeAliasType is required for Pydantic to handle this recursive type correctly.
# Note that TypeAliasType is a backport for Python<3.12, so that when drop Python 3.11 support and
# lower, this can be updated to `type DT = ...`.
# TensorModel must come before dict so Pydantic tries it first during deserialization.
DTModelType = TypeAliasType(
    "DTModelType",
    list["DTModelType"]
    | CompressedTensorModel
    | dict[str, "DTModelType"]
    | str
    | float
    | int
    | bool
    | None,
)
"""Model arbitrary nesting of lists and dicts with typed leaves."""


def _datatree_from(data: DataTree) -> DTModelType:
    """Convert a `DataTree` to a `DTModelType`."""
    if isinstance(data, dict):
        return {k: _datatree_from(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):  # noqa: UP038
        return [_datatree_from(v) for v in data]
    if isinstance(data, np.ndarray):
        return CompressedTensorModel.from_numpy(data)
    return data


def _datatree_to(data: DTModelType) -> DataTree:
    """Convert a `DTModelType` to a `DataTree`."""
    if isinstance(data, CompressedTensorModel):
        return data.to_numpy()
    if isinstance(data, dict):
        return {k: _datatree_to(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_datatree_to(el) for el in data]
    return data


class DataTreeModel(BaseModel):
    """A model of a DataTree."""

    schema_version: Literal["v0.1"] = "v0.1"

    data: DTModelType

    @classmethod
    def from_python(cls, data: DataTree) -> Self:
        """Instantiate from a DataTree."""
        return cls(data=_datatree_from(data))

    def to_python(self) -> DataTree:
        """Convert to a DataTree."""
        return _datatree_to(self.data)
