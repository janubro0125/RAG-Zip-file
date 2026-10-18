from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "data/chroma"

print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

questions = [
    "What are the four dimensions of food security?",
    "What is the population of Japan in 2026?"
]

for question in questions:

    print("\n================================")
    print("QUESTION:", question)
    print("================================")

    results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=4
    )

    for i, (document, score) in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Relevance Score: {score:.4f}")
        print(f"Source: {document.metadata.get('document_name')}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print("Text:")
        print(document.page_content[:300].replace("\n", " "))

print("\nRetrieval test completed.")