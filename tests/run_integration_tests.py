"""
Test runner for integration tests that handles ComfyUI dependencies.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock folder_paths before importing dynamic module
class MockFolderPaths:
    @staticmethod
    def get_output_directory():
        # Return temp directory for tests
        import tempfile
        return tempfile.gettempdir()

sys.modules['folder_paths'] = MockFolderPaths()

# Now we can safely import
import pytest

if __name__ == "__main__":
    # Run the integration tests
    test_file = os.path.join(os.path.dirname(__file__), "test_integration.py")
    exit_code = pytest.main([test_file, "-v", "--tb=short", "-p", "no:warnings"])
    sys.exit(exit_code)
