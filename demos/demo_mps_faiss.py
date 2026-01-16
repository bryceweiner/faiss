#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import os

import importlib.util
from pathlib import Path
import gc

import numpy as np
import torch


def _load_faiss_from(path: Path):
    init_path = path / "__init__.py"
    if not init_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("faiss", init_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["faiss"] = module
    spec.loader.exec_module(module)
    return module


def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "build-mps-py" / "faiss" / "python",
            repo_root / "build-mps" / "faiss" / "python",
            repo_root / "faiss" / "python",
        ]
        for candidate in candidates:
            module = _load_faiss_from(candidate)
            if module is not None:
                return module
        raise


os.environ.setdefault("OS_ACTIVITY_MODE", "disable")
os.environ.setdefault("MTL_DEBUG_LAYER", "0")
os.environ.setdefault("MTL_DEBUG_LAYER_WARNINGS", "0")

faiss = _import_faiss()
from faiss.contrib import torch_utils  # noqa: F401


def _check_mps() -> bool:
    return hasattr(torch, "mps") and torch.backends.mps.is_available()


def _make_data(nb: int, nq: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    xb = torch.randn(nb, d, device="mps", dtype=torch.float32)
    xq = torch.randn(nq, d, device="mps", dtype=torch.float32)
    return xb, xq


def _sync_mps():
    if hasattr(torch, "mps"):
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


def _cpu_reference_flat(xb: np.ndarray, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    cpu_index = faiss.IndexFlatL2(xb.shape[1])
    cpu_index.add(xb)
    D, I = cpu_index.search(xq, k)
    return D, I


def run_flat(nb: int, nq: int, d: int, k: int, seed: int) -> bool:
    xb, xq = _make_data(nb, nq, d, seed)
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlat(res, d, faiss.METRIC_L2)
    index.add(xb)
    D, I = index.search(xq, k)
    _sync_mps()

    D_cpu, I_cpu = _cpu_reference_flat(
        xb.detach().cpu().numpy(), xq.detach().cpu().numpy(), k
    )
    match = np.array_equal(I_cpu, I.cpu().numpy())
    print(f"[flat] top-{k} reference id match: {match} (device={D.device.type})")
    return match


def run_ivfflat(nb: int, nq: int, d: int, k: int, nlist: int, nprobe: int, seed: int) -> bool:
    xb, xq = _make_data(nb, nq, d, seed)
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
    index.nprobe = nprobe
    index.train(xb)
    index.add(xb)
    D, I = index.search(xq, k)
    _sync_mps()

    cpu_quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.nprobe = nprobe
    cpu_xb = xb.detach().cpu().numpy()
    cpu_xq = xq.detach().cpu().numpy()
    cpu_index.train(cpu_xb)
    cpu_index.add(cpu_xb)
    _, I_cpu = cpu_index.search(cpu_xq, k)

    match = np.array_equal(I_cpu, I.cpu().numpy())
    print(f"[ivfflat] top-{k} reference id match: {match} (device={D.device.type})")
    return match


def run_ivfpq(
    nb: int,
    nq: int,
    d: int,
    k: int,
    nlist: int,
    m: int,
    nbits: int,
    nprobe: int,
    seed: int,
) -> bool:
    xb, xq = _make_data(nb, nq, d, seed)
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFPQ(res, d, nlist, m, nbits, faiss.METRIC_L2)
    index.nprobe = nprobe
    index.train(xb)
    index.add(xb)
    D, I = index.search(xq, k)
    _sync_mps()

    cpu_quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    cpu_index.nprobe = nprobe
    cpu_xb = xb.detach().cpu().numpy()
    cpu_xq = xq.detach().cpu().numpy()
    cpu_index.train(cpu_xb)
    cpu_index.add(cpu_xb)
    _, I_cpu = cpu_index.search(cpu_xq, k)

    match = np.array_equal(I_cpu, I.cpu().numpy())
    print(f"[ivfpq] top-{k} reference id match: {match} (device={D.device.type})")
    return match


def main() -> int:
    parser = argparse.ArgumentParser(description="FAISS MPS demo")
    parser.add_argument("--nb", type=int, default=8192)
    parser.add_argument("--nq", type=int, default=256)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=["flat", "ivfflat", "ivfpq", "all"], default="all")
    args = parser.parse_args()

    if not _check_mps():
        print("MPS is not available on this system.")
        return 0
    if args.nb < args.nlist:
        raise ValueError("nb must be >= nlist")
    if args.d % args.m != 0:
        raise ValueError("d must be divisible by m for IVFPQ")
    if args.k > args.nb:
        raise ValueError("k must be <= nb")

    min_train = 39 * max(args.nlist, 1 << args.nbits)
    if args.nb < min_train:
        args.nb = min_train

    print("MPS available:", torch.backends.mps.is_available())
    print("FAISS compile options:", faiss.get_compile_options())

    ok = True
    if args.mode in ("flat", "all"):
        ok = run_flat(args.nb, args.nq, args.d, args.k, args.seed) and ok
    if args.mode in ("ivfflat", "all"):
        ok = run_ivfflat(args.nb, args.nq, args.d, args.k, args.nlist, args.nprobe, args.seed) and ok
    if args.mode in ("ivfpq", "all"):
        ok = run_ivfpq(
            args.nb, args.nq, args.d, args.k,
            args.nlist, args.m, args.nbits, args.nprobe, args.seed
        ) and ok

    print("MPS demo result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
