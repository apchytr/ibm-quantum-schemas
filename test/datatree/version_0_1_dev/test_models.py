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

"""Tests for 0.1 models."""

import numpy as np
import numpy.testing as npt

from ibm_quantum_schemas.datatree.version_0_1_dev import DataTreeModel


def test_initialization():
    """Test initialization for ``DataTreeModel``."""
    passthrough_data = {
        "str": "ciao",
        "float": 1.2,
        "int": 1,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "array": np.array([1.0, 2.0]),
        "nested": {"array2": np.array([3.0, 4.0])},
    }
    model = DataTreeModel.from_python(passthrough_data)
    assert model.schema_version == "v0.1"

    roundtrip = model.to_python()
    npt.assert_equal(roundtrip, passthrough_data)
