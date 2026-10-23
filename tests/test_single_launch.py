"""
tests/test_single_launch.py — Single-Click Launcher Test Suite
--------------------------------------------------------------
Validates:
  1. Service health check detection.
  2. Port availability detection logic.
  3. Canonical agent API endpoint readiness.
  4. Process cleanup handling.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

from launcher import is_port_open, is_backend_healthy, SPAWNED_PROCESSES, cleanup_spawned_processes


class TestSingleLaunch(unittest.TestCase):

    def test_is_port_open_non_existent_port(self):
        """Verify is_port_open returns False for unused ports."""
        # Unlikely to be in use
        self.assertFalse(is_port_open("127.0.0.1", 59999, timeout=0.1))

    @patch("urllib.request.urlopen")
    def test_is_backend_healthy_success(self, mock_urlopen):
        """Verify is_backend_healthy returns True when backend returns status 200."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(is_backend_healthy())

    @patch("urllib.request.urlopen")
    def test_is_backend_healthy_failure(self, mock_urlopen):
        """Verify is_backend_healthy returns False when backend raises exception or non-200."""
        mock_urlopen.side_effect = Exception("Connection refused")
        self.assertFalse(is_backend_healthy())

    def test_cleanup_spawned_processes(self):
        """Verify cleanup_spawned_processes terminates spawned child process mocks."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999

        SPAWNED_PROCESSES.append(mock_proc)

        with patch("sys.exit") as mock_exit:
            cleanup_spawned_processes()
            mock_proc.terminate.assert_called_once()
            mock_exit.assert_called_once_with(0)

        # Clear list after test
        SPAWNED_PROCESSES.clear()


if __name__ == "__main__":
    unittest.main()
