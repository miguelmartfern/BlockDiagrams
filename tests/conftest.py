import pytest

# Hook to configure pytest if needed in the future
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")