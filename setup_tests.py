"""
Setup Tests Directory
Creates test directory structure and basic test files
"""

from pathlib import Path

# Create tests directory
tests_dir = Path("tests")
tests_dir.mkdir(exist_ok=True)

# Create __init__.py
(tests_dir / "__init__.py").write_text('''"""
Test Suite for Book Recommendation System
"""

__version__ = '1.0.0'
''')

# Create conftest.py
(tests_dir / "conftest.py").write_text('''"""
Shared pytest fixtures and configuration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def sample_book_data():
    """Create comprehensive sample book dataset"""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'book_name': [f'Book {i}' for i in range(n)],
        'author': [f'Author {i % 20}' for i in range(n)],
        'rating': np.random.uniform(3.0, 5.0, n),
        'genre': np.random.choice(['Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Fantasy'], n),
        'description': [f'A wonderful book about topic {i % 10}' for i in range(n)],
        'price': np.random.uniform(10.0, 50.0, n),
        'number_of_reviews': np.random.randint(10, 1000, n)
    })


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory):
    """Create temporary data directory"""
    return tmp_path_factory.mktemp("data")


@pytest.fixture
def clean_environment():
    """Clean up test environment"""
    yield
    # Cleanup code here if needed
''')

print("✅ Test directory structure created!")
print("\nCreated files:")
print(f"  - {tests_dir / '__init__.py'}")
print(f"  - {tests_dir / 'conftest.py'}")
