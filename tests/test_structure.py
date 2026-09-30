"""Sanity tests for project structure, module imports, and requirements."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of execution environment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_project_directories():
    """Verify that expected project directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = ["src", "configs", "metadata", "notebooks", "docs"]
    for d in expected_dirs:
        assert (base_dir / d).is_dir(), f"Expected directory {d} missing"

def test_src_imports():
    """Verify that src submodules can be imported cleanly."""
    import src.data
    import src.preprocessing
    import src.models
    import src.training
    import src.evaluation

    assert src.data is not None
    assert src.preprocessing is not None
    assert src.models is not None
    assert src.training is not None
    assert src.evaluation is not None

def test_metadata_file_exists():
    """Verify metadata directory contains expected schema docs."""
    base_dir = Path(__file__).resolve().parent.parent
    readme_path = base_dir / "metadata" / "README.md"
    assert readme_path.is_file(), "Metadata README is missing"
