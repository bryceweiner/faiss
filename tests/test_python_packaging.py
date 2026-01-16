import pathlib
import unittest


class TestPythonPackaging(unittest.TestCase):
    def test_pyproject_requires_numpy(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        pyproject = repo_root / "faiss" / "python" / "pyproject.toml"
        contents = pyproject.read_text(encoding="utf-8")
        self.assertIn('numpy', contents)

    def test_setup_py_uses_python_hints(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        setup_py = repo_root / "faiss" / "python" / "setup.py"
        contents = setup_py.read_text(encoding="utf-8")
        self.assertIn("Python_EXECUTABLE", contents)
        self.assertIn("_find_python_source", contents)
        self.assertIn("_find_python_source_for_lib", contents)
        self.assertIn("_copy_extension", contents)
        self.assertIn("(build_dir, root_dir)", contents)

    def test_swigfaiss_mps_gpu_vector_guard(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        swig = repo_root / "faiss" / "python" / "swigfaiss.swig"
        contents = swig.read_text(encoding="utf-8")
        self.assertIn("%template(GpuResourcesVector)", contents)
        self.assertIn("FAISS_ENABLE_MPS", contents)


if __name__ == "__main__":
    unittest.main()
