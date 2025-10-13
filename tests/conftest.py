from pathlib import Path

def pytest_ignore_collect(collection_path: Path):
    """Exclude the comprehensive script-like file from test collection."""
    return collection_path.name == "test_all_indicators_comprehensive.py"
