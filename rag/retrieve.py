from langchain_community.embeddings import HuggingFaceEmbeddings  # ← updated
from langchain_chroma import Chroma                        # ← updated

CHROMA_PATH = "chroma_db"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    return retriever

if __name__ == "__main__":
    retriever = get_retriever()
    query = input("Enter your question: ")
    results = retriever.invoke(query)
    print("\nTop Retrieved Chunks:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:\n")
        print(doc.page_content)
        print("\n" + "-"*50 + "\n")
        
