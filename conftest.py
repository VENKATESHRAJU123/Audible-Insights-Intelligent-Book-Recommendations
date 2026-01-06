
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
