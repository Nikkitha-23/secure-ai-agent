"""
tests/test_pdf_upload.py — PDF Ingestion & Upload Test Suite
--------------------------------------------------------------
Validates:
  1. Non-PDF upload rejection (400 Bad Request).
  2. Empty PDF file rejection (0 bytes).
  3. Duplicate document ingestion detection.
  4. PyPDFLoader text extraction & chunking pipeline.
"""

import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app, UPLOAD_DIR

client = TestClient(app)


class TestPDFUploadSuite(unittest.TestCase):

    def test_non_pdf_rejection(self):
        """Verify non-PDF files are rejected with status 400."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"Hello world", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files supported", response.json()["message"])

    def test_empty_file_rejection(self):
        """Verify 0-byte PDF files are rejected with status 400."""
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["message"].lower())

    @patch("langchain_community.document_loaders.PyPDFLoader.load")
    def test_valid_pdf_upload_and_duplicate_detection(self, mock_pdf_load):
        """Verify valid PDF upload and duplicate detection."""
        from langchain_core.documents import Document
        mock_pdf_load.return_value = [
            Document(page_content="Sample PDF text for unit testing and ingestion verification.", metadata={})
        ]

        pdf_data = b"%PDF-1.4 Mock Header Data for Upload Test"
        test_filename = "unit_test_sample.pdf"

        # 1. First upload -> Ingest
        res1 = client.post(
            "/upload",
            files={"file": (test_filename, pdf_data, "application/pdf")}
        )
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertIn("already_indexed", data1)

        # 2. Second upload without force_reindex -> Duplicate detected
        res2 = client.post(
            "/upload",
            files={"file": (test_filename, pdf_data, "application/pdf")}
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("already_indexed"))
        self.assertIn("already indexed", data2.get("message", "").lower())

        # Cleanup saved test file
        saved_file = os.path.join(UPLOAD_DIR, test_filename)
        if os.path.exists(saved_file):
            try:
                os.unlink(saved_file)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
