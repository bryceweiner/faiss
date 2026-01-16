# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from demos import demo_mps_faiss


@pytest.mark.skipif(
    not hasattr(torch, "mps") or not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_mps_demo_flat():
    ok = demo_mps_faiss.run_flat(nb=64, nq=8, d=16, k=3, seed=42)
    assert ok
