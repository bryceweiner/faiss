# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil
import subprocess
from pathlib import Path
import sys
import sysconfig

from setuptools import setup

def _resolve_repo_root():
    root_dir = Path(__file__).resolve().parent
    for candidate in root_dir.parents:
        if (candidate / "contrib").exists():
            return candidate
    return root_dir.parents[1]


def _ensure_contrib(repo_root: Path, root_dir: Path) -> Path:
    contrib_path = repo_root / "contrib"
    if contrib_path.exists():
        return contrib_path
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root_dir), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
        top_path = Path(top)
        subprocess.run(
            ["git", "-C", str(top_path), "checkout", "HEAD", "--", "contrib"],
            check=False,
        )
        contrib_path = top_path / "contrib"
        if contrib_path.exists():
            return contrib_path
    except Exception:
        pass
    raise FileNotFoundError("Could not locate 'contrib' directory for packaging.")


def _run_cmake_build():
    repo_root = _resolve_repo_root()
    build_dir = repo_root / "build-pip"
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        env.setdefault("FAISS_ENABLE_GPU", "ON")
        env.setdefault("FAISS_ENABLE_MPS", "ON")
        env.setdefault(
            "OpenMP_C_FLAGS",
            "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include",
        )
        env.setdefault(
            "OpenMP_CXX_FLAGS",
            "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include",
        )
        env.setdefault("OpenMP_C_LIB_NAMES", "omp")
        env.setdefault("OpenMP_CXX_LIB_NAMES", "omp")
        env.setdefault(
            "OpenMP_omp_LIBRARY", "/opt/homebrew/opt/libomp/lib/libomp.dylib"
        )

    cmake_args = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DFAISS_ENABLE_PYTHON=ON",
        "-DBUILD_TESTING=OFF",
    ]
    cmake_args.append(f"-DPython_EXECUTABLE={sys.executable}")
    cmake_args.append(f"-DPython3_EXECUTABLE={sys.executable}")
    python_include = sysconfig.get_paths().get("include")
    if python_include:
        cmake_args.append(f"-DPython_INCLUDE_DIRS={python_include}")
        cmake_args.append(f"-DPython3_INCLUDE_DIRS={python_include}")
    python_libdir = sysconfig.get_config_var("LIBDIR")
    python_ldlib = sysconfig.get_config_var("LDLIBRARY")
    if python_libdir and python_ldlib:
        python_lib = str(Path(python_libdir) / python_ldlib)
        cmake_args.append(f"-DPython_LIBRARY={python_lib}")
        cmake_args.append(f"-DPython3_LIBRARY={python_lib}")
    python_root = sysconfig.get_config_var("prefix") or sys.base_prefix
    if python_root:
        cmake_args.append(f"-DPython_ROOT_DIR={python_root}")
        cmake_args.append(f"-DPython3_ROOT_DIR={python_root}")
    try:
        import numpy

        numpy_include = numpy.get_include()
        cmake_args.append(f"-DPython_NumPy_INCLUDE_DIRS={numpy_include}")
        cmake_args.append(f"-DPython3_NumPy_INCLUDE_DIRS={numpy_include}")
    except Exception:
        pass
    if env.get("FAISS_ENABLE_GPU"):
        cmake_args.append(f"-DFAISS_ENABLE_GPU={env['FAISS_ENABLE_GPU']}")
    if env.get("FAISS_ENABLE_MPS"):
        cmake_args.append(f"-DFAISS_ENABLE_MPS={env['FAISS_ENABLE_MPS']}")
    if env.get("OpenMP_C_FLAGS"):
        cmake_args.append(f"-DOpenMP_C_FLAGS={env['OpenMP_C_FLAGS']}")
    if env.get("OpenMP_CXX_FLAGS"):
        cmake_args.append(f"-DOpenMP_CXX_FLAGS={env['OpenMP_CXX_FLAGS']}")
    if env.get("OpenMP_C_LIB_NAMES"):
        cmake_args.append(f"-DOpenMP_C_LIB_NAMES={env['OpenMP_C_LIB_NAMES']}")
    if env.get("OpenMP_CXX_LIB_NAMES"):
        cmake_args.append(f"-DOpenMP_CXX_LIB_NAMES={env['OpenMP_CXX_LIB_NAMES']}")
    if env.get("OpenMP_omp_LIBRARY"):
        cmake_args.append(f"-DOpenMP_omp_LIBRARY={env['OpenMP_omp_LIBRARY']}")

    subprocess.run(cmake_args, check=True, env=env)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        check=True,
        env=env,
    )


def _candidate_lib_dirs():
    root_dir = Path(__file__).resolve().parent
    repo_root = _resolve_repo_root()
    build_dir = repo_root / "build-pip" / "faiss" / "python"
    return [build_dir, root_dir]


def _find_lib(name):
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for base in _candidate_lib_dirs():
        candidate = base / name
        if candidate.exists():
            return str(candidate)
        if ext_suffix and name.endswith(ext) and ext_suffix != ext:
            alt_name = name[: -len(ext)] + ext_suffix
            candidate = base / alt_name
            if candidate.exists():
                return str(candidate)
    return None


def _find_python_source(name, preferred_dir=None):
    root_dir = Path(__file__).resolve().parent
    repo_root = _resolve_repo_root()
    build_dir = repo_root / "build-pip" / "faiss" / "python"
    search_dirs = []
    if preferred_dir is not None:
        search_dirs.append(Path(preferred_dir))
    search_dirs.extend([build_dir, root_dir])
    for base in search_dirs:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _find_python_source_for_lib(lib_path, name):
    if lib_path:
        lib_dir = Path(lib_path).resolve().parent
        candidate = _find_python_source(name, preferred_dir=lib_dir)
        if candidate:
            return candidate
    return _find_python_source(name)


def _copy_extension(src_path, dest_dir, module_basename, ext_suffix):
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dest_dir / f"{module_basename}{ext}")
    if ext_suffix and ext_suffix != ext:
        shutil.copyfile(src_path, dest_dir / f"{module_basename}{ext_suffix}")


root_dir = Path(__file__).resolve().parent
repo_root = _resolve_repo_root()
contrib_src = _ensure_contrib(repo_root, root_dir)
shutil.rmtree(root_dir / "faiss", ignore_errors=True)
os.mkdir(root_dir / "faiss")
shutil.copytree(contrib_src, root_dir / "faiss" / "contrib")
shutil.copyfile(root_dir / "__init__.py", root_dir / "faiss" / "__init__.py")
shutil.copyfile(root_dir / "loader.py", root_dir / "faiss" / "loader.py")
shutil.copyfile(
    root_dir / "class_wrappers.py", root_dir / "faiss" / "class_wrappers.py"
)
shutil.copyfile(root_dir / "gpu_wrappers.py", root_dir / "faiss" / "gpu_wrappers.py")
shutil.copyfile(
    root_dir / "extra_wrappers.py", root_dir / "faiss" / "extra_wrappers.py"
)
shutil.copyfile(
    root_dir / "array_conversions.py", root_dir / "faiss" / "array_conversions.py"
)

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
prefix = "Release/" * (platform.system() == "Windows")

swigfaiss_generic_lib = f"{prefix}_swigfaiss{ext}"
swigfaiss_avx2_lib = f"{prefix}_swigfaiss_avx2{ext}"
swigfaiss_avx512_lib = f"{prefix}_swigfaiss_avx512{ext}"
swigfaiss_avx512_spr_lib = f"{prefix}_swigfaiss_avx512_spr{ext}"
callbacks_lib = f"{prefix}libfaiss_python_callbacks{ext}"
swigfaiss_sve_lib = f"{prefix}_swigfaiss_sve{ext}"
faiss_example_external_module_lib = f"_faiss_example_external_module{ext}"

swigfaiss_generic_path = _find_lib(swigfaiss_generic_lib)
swigfaiss_avx2_path = _find_lib(swigfaiss_avx2_lib)
swigfaiss_avx512_path = _find_lib(swigfaiss_avx512_lib)
swigfaiss_avx512_spr_path = _find_lib(swigfaiss_avx512_spr_lib)
callbacks_path = _find_lib(callbacks_lib)
swigfaiss_sve_path = _find_lib(swigfaiss_sve_lib)
faiss_example_external_module_path = _find_lib(faiss_example_external_module_lib)

found_swigfaiss_generic = swigfaiss_generic_path is not None
found_swigfaiss_avx2 = swigfaiss_avx2_path is not None
found_swigfaiss_avx512 = swigfaiss_avx512_path is not None
found_swigfaiss_avx512_spr = swigfaiss_avx512_spr_path is not None
found_callbacks = callbacks_path is not None
found_swigfaiss_sve = swigfaiss_sve_path is not None
found_faiss_example_external_module_lib = faiss_example_external_module_path is not None

if platform.system() != "AIX":
    if not (
        found_swigfaiss_generic
        or found_swigfaiss_avx2
        or found_swigfaiss_avx512
        or found_swigfaiss_avx512_spr
        or found_swigfaiss_sve
        or found_faiss_example_external_module_lib
    ):
        _run_cmake_build()
        swigfaiss_generic_path = _find_lib(swigfaiss_generic_lib)
        swigfaiss_avx2_path = _find_lib(swigfaiss_avx2_lib)
        swigfaiss_avx512_path = _find_lib(swigfaiss_avx512_lib)
        swigfaiss_avx512_spr_path = _find_lib(swigfaiss_avx512_spr_lib)
        callbacks_path = _find_lib(callbacks_lib)
        swigfaiss_sve_path = _find_lib(swigfaiss_sve_lib)
        faiss_example_external_module_path = _find_lib(faiss_example_external_module_lib)

        found_swigfaiss_generic = swigfaiss_generic_path is not None
        found_swigfaiss_avx2 = swigfaiss_avx2_path is not None
        found_swigfaiss_avx512 = swigfaiss_avx512_path is not None
        found_swigfaiss_avx512_spr = swigfaiss_avx512_spr_path is not None
        found_callbacks = callbacks_path is not None
        found_swigfaiss_sve = swigfaiss_sve_path is not None
        found_faiss_example_external_module_lib = faiss_example_external_module_path is not None

    assert (
        found_swigfaiss_generic
        or found_swigfaiss_avx2
        or found_swigfaiss_avx512
        or found_swigfaiss_avx512_spr
        or found_swigfaiss_sve
        or found_faiss_example_external_module_lib
    ), (
        f"Could not find {swigfaiss_generic_lib} or "
        f"{swigfaiss_avx2_lib} or {swigfaiss_avx512_lib} or {swigfaiss_avx512_spr_lib} or {swigfaiss_sve_lib} or {faiss_example_external_module_lib}. "
        f"Faiss may not be compiled yet."
    )

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_path}")
    swigfaiss_py = _find_python_source_for_lib(
        swigfaiss_generic_path, "swigfaiss.py"
    )
    if not swigfaiss_py:
        raise FileNotFoundError("Could not locate swigfaiss.py after build.")
    shutil.copyfile(swigfaiss_py, root_dir / "faiss" / "swigfaiss.py")
    _copy_extension(
        swigfaiss_generic_path, root_dir / "faiss", "_swigfaiss", ext_suffix
    )

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_path}")
    swigfaiss_avx2_py = _find_python_source_for_lib(
        swigfaiss_avx2_path, "swigfaiss_avx2.py"
    )
    if not swigfaiss_avx2_py:
        raise FileNotFoundError("Could not locate swigfaiss_avx2.py after build.")
    shutil.copyfile(swigfaiss_avx2_py, root_dir / "faiss" / "swigfaiss_avx2.py")
    _copy_extension(
        swigfaiss_avx2_path, root_dir / "faiss", "_swigfaiss_avx2", ext_suffix
    )

if found_swigfaiss_avx512:
    print(f"Copying {swigfaiss_avx512_path}")
    swigfaiss_avx512_py = _find_python_source_for_lib(
        swigfaiss_avx512_path, "swigfaiss_avx512.py"
    )
    if not swigfaiss_avx512_py:
        raise FileNotFoundError("Could not locate swigfaiss_avx512.py after build.")
    shutil.copyfile(swigfaiss_avx512_py, root_dir / "faiss" / "swigfaiss_avx512.py")
    _copy_extension(
        swigfaiss_avx512_path, root_dir / "faiss", "_swigfaiss_avx512", ext_suffix
    )

if found_swigfaiss_avx512_spr:
    print(f"Copying {swigfaiss_avx512_spr_path}")
    swigfaiss_avx512_spr_py = _find_python_source_for_lib(
        swigfaiss_avx512_spr_path, "swigfaiss_avx512_spr.py"
    )
    if not swigfaiss_avx512_spr_py:
        raise FileNotFoundError(
            "Could not locate swigfaiss_avx512_spr.py after build."
        )
    shutil.copyfile(
        swigfaiss_avx512_spr_py,
        root_dir / "faiss" / "swigfaiss_avx512_spr.py",
    )
    _copy_extension(
        swigfaiss_avx512_spr_path,
        root_dir / "faiss",
        "_swigfaiss_avx512_spr",
        ext_suffix,
    )

if found_callbacks:
    print(f"Copying {callbacks_path}")
    shutil.copyfile(callbacks_path, root_dir / "faiss" / callbacks_lib)

if found_swigfaiss_sve:
    print(f"Copying {swigfaiss_sve_path}")
    swigfaiss_sve_py = _find_python_source_for_lib(
        swigfaiss_sve_path, "swigfaiss_sve.py"
    )
    if not swigfaiss_sve_py:
        raise FileNotFoundError("Could not locate swigfaiss_sve.py after build.")
    shutil.copyfile(
        swigfaiss_sve_py,
        root_dir / "faiss" / "swigfaiss_sve.py",
    )
    _copy_extension(
        swigfaiss_sve_path, root_dir / "faiss", "_swigfaiss_sve", ext_suffix
    )

if found_faiss_example_external_module_lib:
    print(f"Copying {faiss_example_external_module_path}")
    example_py = _find_python_source_for_lib(
        faiss_example_external_module_path, "faiss_example_external_module.py"
    )
    if not example_py:
        raise FileNotFoundError(
            "Could not locate faiss_example_external_module.py after build."
        )
    shutil.copyfile(
        example_py,
        root_dir / "faiss" / "faiss_example_external_module.py",
    )
    _copy_extension(
        faiss_example_external_module_path,
        root_dir / "faiss",
        "_faiss_example_external_module",
        ext_suffix,
    )

long_description = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
setup(
    name="faiss",
    version="1.13.2",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging"],
    packages=["faiss", "faiss.contrib", "faiss.contrib.torch"],
    package_data={
        "faiss": ["*.so", "*.pyd", "*.a"],
    },
    zip_safe=False,
)