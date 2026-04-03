"""
ingest_healthcare_docs.py — Secure AI Agent
Ingests healthcare PDFs into ChromaDB
Source: data/healthcare/
Files: Annual-Report-FY2023-2024.pdf, patient_Guide.pdf
Collection: healthcare_general
"""

import os
import uuid
import chromadb
from pypdf import PdfReader

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HEALTHCARE_DIR   = "data/healthcare"
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "healthcare_general"
CHUNK_SIZE       = 500   # characters per chunk
CHUNK_OVERLAP    = 50    # overlap between chunks
TENANT           = "apollo_hospital"

# ─────────────────────────────────────────────
# PDF READER
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


# ─────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def ingest_healthcare_docs():
    """Main ingestion function."""

    # Init ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if present (fresh ingest)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(COLLECTION_NAME)
    print(f"✅ Created collection: {COLLECTION_NAME}\n")

    # Find PDFs
    pdf_files = [f for f in os.listdir(HEALTHCARE_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found in data/healthcare/")
        return

    total_chunks = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(HEALTHCARE_DIR, pdf_file)
        print(f"📄 Processing: {pdf_file}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"   ⚠️  No text extracted from {pdf_file}, skipping.")
            continue

        print(f"   📝 Extracted {len(text)} characters")

        # Chunk
        chunks = chunk_text(text)
        print(f"   🔪 Split into {len(chunks)} chunks")

        # Prepare for ChromaDB
        ids        = [str(uuid.uuid4()) for _ in chunks]
        metadatas  = [
            {
                "source":    pdf_file,
                "tenant":    TENANT,
                "domain":    "healthcare",
                "chunk_idx": i,
            }
            for i in range(len(chunks))
        ]

        # Add to collection in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                documents=chunks[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        total_chunks += len(chunks)
        print(f"   ✅ Ingested {len(chunks)} chunks\n")

    print("=" * 50)
    print(f"✅ Ingestion complete!")
    print(f"   Collection : {COLLECTION_NAME}")
    print(f"   Tenant     : {TENANT}")
    print(f"   Total chunks: {total_chunks}")
    print("=" * 50)


# ─────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────

def verify_collection():
    """Quick check — how many docs in collection?"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"\n🔍 Verification: {COLLECTION_NAME} → {count} chunks")
    except Exception as e:
        print(f"❌ Collection not found: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Secure AI Agent — Healthcare Ingestion")
    print("=" * 50)
    print(f"  Source : {HEALTHCARE_DIR}")
    print(f"  DB     : {CHROMA_DB_PATH}")
    print(f"  Collection: {COLLECTION_NAME}")
    print("=" * 50 + "\n")

    ingest_healthcare_docs()
    verify_collection()