"""
ingest_education_docs.py
------------------------
Loads education PDFs into ChromaDB with multi-tenant isolation.
Each tenant gets its own collection: education_{tenant_id}

Usage:
    python ingest_education_docs.py
    python ingest_education_docs.py --data_dir ./data/education --chroma_dir ./chroma_db
"""

import os
import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ─── TENANT MAP ───────────────────────────────────────────────────────────────
# Map filename keywords → tenant_id
# Edit this if you add more PDFs later

TENANT_MAP = {
    "B.Tech.AIDS":                    "anna_university",
    "BE CSE":                         "anna_university",
    "CEG_UG_Fee_Structure":           "anna_university",
    "MIT":                            "mit",
    "ACADEMIC REGULATIONS":           "other",
    "Artificial_Intelligence":        "other",
    "IJRTI2304061":                   "other",
}

def get_tenant(filename: str) -> str:
    """Match filename to tenant using keyword map."""
    for keyword, tenant in TENANT_MAP.items():
        if keyword.lower() in filename.lower():
            return tenant
    return "other"  # default fallback


# ─── PDF → CHUNKS ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # overlap for context continuity
    return chunks


def make_chunk_id(filename: str, chunk_index: int) -> str:
    """Stable unique ID for each chunk."""
    raw = f"{filename}__chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ─── INGEST ───────────────────────────────────────────────────────────────────

def ingest_all(data_dir: str, chroma_dir: str):
    pdf_dir = Path(data_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[!] No PDFs found in {data_dir}")
        return

    # Init ChromaDB (persistent)
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    stats = {}  # tenant_id → chunk count

    for pdf_path in sorted(pdf_files):
        filename = pdf_path.name
        tenant_id = get_tenant(filename)
        collection_name = f"education_{tenant_id}"

        print(f"\n📄 {filename}")
        print(f"   └── Tenant  : {tenant_id}")
        print(f"   └── Collection: {collection_name}")

        # Extract text
        try:
            text = extract_text_from_pdf(str(pdf_path))
        except Exception as e:
            print(f"   [ERROR] Could not read PDF: {e}")
            continue

        if not text:
            print(f"   [WARN] Empty text — skipping")
            continue

        # Chunk
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        print(f"   └── Chunks  : {len(chunks)}")

        # Get or create tenant collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "tenant_id": tenant_id,
                "domain": "education",
                "hnsw:space": "cosine"
            }
        )

        # Upsert chunks (idempotent — safe to re-run)
        ids = [make_chunk_id(filename, i) for i in range(len(chunks))]
        metadatas = [
            {
                "tenant_id": tenant_id,
                "source_file": filename,
                "chunk_index": i,
                "domain": "education",
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        stats[tenant_id] = stats.get(tenant_id, 0) + len(chunks)
        print(f"   └── ✅ Upserted into '{collection_name}'")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("✅ INGESTION COMPLETE")
    print("=" * 50)
    for tenant, count in stats.items():
        print(f"  {tenant:25s} → {count} chunks")

    # List all collections
    print("\n📦 ChromaDB Collections:")
    for col in client.list_collections():
        print(f"  - {col.name}  ({col.count()} docs)")


# ─── VERIFY: Test tenant isolation ────────────────────────────────────────────

def verify_isolation(chroma_dir: str):
    """Quick sanity check — query one tenant, confirm no cross-tenant leakage."""
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    print("\n🔍 ISOLATION VERIFICATION")
    print("-" * 40)

    collections = client.list_collections()
    if not collections:
        print("[!] No collections found. Run ingestion first.")
        return

    for col in collections:
        collection = client.get_collection(col.name)
        results = collection.query(
            query_texts=["What is the attendance policy?"],
            n_results=min(2, collection.count()),
            include=["documents", "metadatas"]
        )

        print(f"\nCollection: {col.name}")
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            tenant_in_result = meta.get("tenant_id", "unknown")
            expected_tenant = col.name.replace("education_", "")
            status = "✅" if tenant_in_result == expected_tenant else "❌ LEAK!"
            print(f"  Result {i+1}: tenant={tenant_in_result} {status}")
            print(f"  Preview : {doc[:120]}...")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest education PDFs into ChromaDB")
    parser.add_argument("--data_dir",   default="./data/education", help="Folder with PDFs")
    parser.add_argument("--chroma_dir", default="./chroma_db",      help="ChromaDB storage path")
    parser.add_argument("--verify",     action="store_true",         help="Run isolation check after ingestion")
    args = parser.parse_args()

    os.makedirs(args.chroma_dir, exist_ok=True)

    ingest_all(args.data_dir, args.chroma_dir)

    if args.verify:
        verify_isolation(args.chroma_dir)