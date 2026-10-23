"""
FILE: tools/file_reader_tool.py
--------------------------------
File Reader Tool - Agent-callable implementation
Supports: PDF, DOCX, TXT, CSV, XLSX, images (OCR)
"""

import os
import json
from pathlib import Path
from typing import Any


UPLOAD_DIR = "uploads"  # change to your actual upload path


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read and extract text content from an uploaded file. "
            "Supports PDF, DOCX, TXT, CSV, XLSX formats. "
            "Use this when the user asks about a document they have uploaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the uploaded file (e.g. 'report.pdf', 'notes.txt')"
                },
                "page_range": {
                    "type": "string",
                    "description": "Optional. Page range to read for PDFs, e.g. '1-5' or '3'. Default: all pages."
                }
            },
            "required": ["filename"]
        }
    }
}


def read_file(filename: str, page_range: str = None) -> dict[str, Any]:
    """
    Main tool function called by the agent.
    Returns dict with 'success', 'content', and 'metadata'.
    """
    filepath = Path(UPLOAD_DIR) / filename

    if not filepath.exists():
        return {
            "success": False,
            "error": f"File '{filename}' not found in uploads folder.",
            "available_files": _list_uploads()
        }

    suffix = filepath.suffix.lower()

    try:
        if suffix == ".txt":
            content = _read_txt(filepath)
        elif suffix == ".pdf":
            content = _read_pdf(filepath, page_range)
        elif suffix == ".docx":
            content = _read_docx(filepath)
        elif suffix == ".csv":
            content = _read_csv(filepath)
        elif suffix in (".xlsx", ".xls"):
            content = _read_excel(filepath)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: '{suffix}'. Supported: .txt, .pdf, .docx, .csv, .xlsx"
            }

        return {
            "success": True,
            "filename": filename,
            "file_type": suffix,
            "content": content[:8000],          # safety: cap at 8k chars to fit context window
            "truncated": len(content) > 8000,
            "total_chars": len(content)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read '{filename}': {str(e)}"
        }


def _read_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="replace")


def _read_pdf(filepath: Path, page_range: str = None) -> str:
    """Uses pypdf (pip install pypdf)"""
    try:
        import pypdf
    except ImportError:
        return "[ERROR] pypdf not installed. Run: pip install pypdf"

    reader = pypdf.PdfReader(str(filepath))
    pages = reader.pages

    # parse optional page_range e.g. "1-5" or "3"
    if page_range:
        try:
            if "-" in page_range:
                start, end = page_range.split("-")
                pages = reader.pages[int(start)-1 : int(end)]
            else:
                pages = [reader.pages[int(page_range)-1]]
        except Exception:
            pass  # fall back to all pages

    return "\n\n".join(p.extract_text() or "" for p in pages)


def _read_docx(filepath: Path) -> str:
    """Uses python-docx (pip install python-docx)"""
    try:
        import docx
    except ImportError:
        return "[ERROR] python-docx not installed. Run: pip install python-docx"

    doc = docx.Document(str(filepath))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_csv(filepath: Path) -> str:
    import csv
    rows = []
    with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(", ".join(row))
    return "\n".join(rows)


def _read_excel(filepath: Path) -> str:
    """Uses openpyxl (pip install openpyxl)"""
    try:
        import openpyxl
    except ImportError:
        return "[ERROR] openpyxl not installed. Run: pip install openpyxl"

    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    result = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        result.append(f"=== Sheet: {sheet} ===")
        for row in ws.iter_rows(values_only=True):
            result.append(", ".join(str(c) if c is not None else "" for c in row))
    return "\n".join(result)


def _list_uploads() -> list[str]:
    upload_path = Path(UPLOAD_DIR)
    if not upload_path.exists():
        return []
    return [f.name for f in upload_path.iterdir() if f.is_file()]
