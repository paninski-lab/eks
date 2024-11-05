import pytest
import sys

def main():
    # Get arguments from the command line (excluding the script name)
    args = sys.argv[1:]

    # Default to running both test files if no arguments are provided
    if not args:
        test_files = ["test_core.py", "test_singlecam_smoother.py"]
    else:
        # Use provided arguments as the list of test files to run
        test_files = args

    # Run pytest on the specified test files
    result = pytest.main(["-v"] + test_files)
    if result == 0:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")

if __name__ == "__main__":
    main()