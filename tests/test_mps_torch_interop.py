# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import importlib.util
import pathlib
import sys


def _load_local_faiss_package():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build" / "faiss" / "python"
    src_dir = repo_root / "faiss" / "python"
    if not build_dir.joinpath("swigfaiss.py").exists():
        return None
    sys.path.insert(0, str(build_dir))
    pkg = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("faiss", loader=None)
    )
    pkg.__path__ = [str(src_dir), str(build_dir)]
    sys.modules["faiss"] = pkg
    import swigfaiss as _swigfaiss
    for name in dir(_swigfaiss):
        if not name.startswith("__"):
            setattr(pkg, name, getattr(_swigfaiss, name))
    return pkg


faiss_pkg = _load_local_faiss_package()
if faiss_pkg is None:
    import faiss  # noqa: E402
else:
    import importlib
    importlib.import_module("faiss.__init__")
    import faiss  # noqa: E402

# Ensure class wrappers are applied for swigfaiss-only imports
import inspect
from faiss import class_wrappers  # noqa: E402

for symbol in dir(faiss):
    obj = getattr(faiss, symbol)
    if inspect.isclass(obj):
        if issubclass(obj, faiss.Index):
            class_wrappers.handle_Index(obj)
        if issubclass(obj, faiss.IndexBinary):
            class_wrappers.handle_IndexBinary(obj)
        if issubclass(obj, faiss.VectorTransform):
            class_wrappers.handle_VectorTransform(obj)
        if issubclass(obj, faiss.Quantizer):
            class_wrappers.handle_Quantizer(obj)

# Import torch_utils after faiss is set
from faiss.contrib import torch_utils  # noqa: F401,E402


@pytest.mark.skipif(
    not hasattr(torch, "mps") or not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_mps_index_flat_add_search():
    d = 4
    nb = 8
    nq = 3
    k = 2

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlat(res, d, faiss.METRIC_L2)

    xb = torch.randn(nb, d, device="mps", dtype=torch.float32)
    index.add(xb)

    xq = torch.randn(nq, d, device="mps", dtype=torch.float32)
    D, I = index.search(xq, k)

    assert D.device.type == "mps"
    assert I.device.type == "mps"
    assert D.shape == (nq, k)
    assert I.shape == (nq, k)


@pytest.mark.skipif(
    not hasattr(torch, "mps") or not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_mps_index_ivf_flat_add_search():
    d = 8
    nb = 64
    nq = 4
    nlist = 8
    k = 3

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
    xb = torch.randn(nb, d, device="mps", dtype=torch.float32)
    index.train(xb)
    index.add(xb)
    index.nprobe = 4

    xq = torch.randn(nq, d, device="mps", dtype=torch.float32)
    D, I = index.search(xq, k)

    assert D.device.type == "mps"
    assert I.device.type == "mps"
    assert D.shape == (nq, k)
    assert I.shape == (nq, k)


@pytest.mark.skipif(
    not hasattr(torch, "mps") or not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_mps_index_ivf_pq_add_search():
    d = 8
    nb = 64
    nq = 4
    nlist = 8
    m = 2
    nbits = 4
    k = 3

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFPQ(res, d, nlist, m, nbits, faiss.METRIC_L2)
    xb = torch.randn(nb, d, device="mps", dtype=torch.float32)
    index.train(xb)
    index.add(xb)
    index.nprobe = 4

    xq = torch.randn(nq, d, device="mps", dtype=torch.float32)
    D, I = index.search(xq, k)

    assert D.device.type == "mps"
    assert I.device.type == "mps"
    assert D.shape == (nq, k)
    assert I.shape == (nq, k)
