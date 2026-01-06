"""
Complete test setup and verification
"""

from pathlib import Path
import subprocess
import sys

def setup_tests():
    """Setup test directory and files"""
    
    print("Setting up tests directory...")
    
    # Create tests directory
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    (tests_dir / "__init__.py").write_text('"""Test Suite"""')
    
    # Create conftest.py with fixtures
    (tests_dir / "conftest.py").write_text('''
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_book_data():
    np.random.seed(42)
    return pd.DataFrame({
        'book_name': [f'Book {i}' for i in range(100)],
        'author': [f'Author {i % 20}' for i in range(100)],
        'rating': np.random.uniform(3.0, 5.0, 100),
        'genre': np.random.choice(['Fiction', 'Mystery'], 100),
        'price': np.random.uniform(10.0, 50.0, 100)
    })
''')
    
    # Create basic test file
    (tests_dir / "test_basic.py").write_text('''
import pytest

def test_basic():
    """Basic test"""
    assert 1 + 1 == 2

def test_imports():
    """Test imports work"""
    import pandas as pd
    import numpy as np
    assert pd is not None
    assert np is not None

class TestBasic:
    def test_example(self):
        assert True
''')
    
    print("✅ Test directory created!")
    print(f"   - {tests_dir / '__init__.py'}")
    print(f"   - {tests_dir / 'conftest.py'}")
    print(f"   - {tests_dir / 'test_basic.py'}")
    
    # Run basic test
    print("\n" + "="*60)
    print("Running basic tests...")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_basic.py", "-v"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✅ Tests are working!")
        print("\nNext steps:")
        print("1. Add full test files (I can provide them)")
        print("2. Run: pytest -v")
        print("3. Run with coverage: pytest --cov=src")
    else:
        print("\n❌ Tests failed. Check the output above.")

if __name__ == "__main__":
    setup_tests()
