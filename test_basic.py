
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
