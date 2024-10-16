import sys

def test_basic_functionality():
    assert 2 + 2 == 4, "Basic math failed!"

def test_python_version():
    assert sys.version_info >= (3, 11), "Python version is less than 3.8!"

if __name__ == "__main__":
    test_basic_functionality()
    test_python_version()
    print("Smoke test passed!")
