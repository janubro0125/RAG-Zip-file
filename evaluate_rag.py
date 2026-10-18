import os
import warnings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings(
    "ignore",
    message="Relevance scores must be between 0 and 1"
)

CHROMA_DIR = "data/chroma"
THRESHOLD = 0.40


print("================================")
print("       AETHRIX RAG EVALUATION")
print("================================")


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("\nLoading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# LOAD CHROMADB
# ==========================================

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)


# ==========================================
# HELPER FUNCTION
# ==========================================

def retrieve(question, k=4):

    results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    accepted = []

    for document, score in results:

        if score >= THRESHOLD:
            accepted.append((document, score))

    return results, accepted


# ==========================================
# TEST 1 — KNOWN ANSWER
# ==========================================

print("\n================================")
print("TEST 1 — KNOWN ANSWER")
print("================================")

question = "What are the four dimensions of food security?"

results, accepted = retrieve(question)

if accepted:
    print("PASS: Relevant documents retrieved.")
    print(f"Accepted results: {len(accepted)}")
else:
    print("FAIL: No relevant documents retrieved.")


# ==========================================
# TEST 2 — CROSS DOCUMENT
# ==========================================

print("\n================================")
print("TEST 2 — CROSS DOCUMENT")
print("================================")

question = "What are the major challenges facing food and agriculture?"

results, accepted = retrieve(question)

documents_found = set()

for document, score in accepted:

    documents_found.add(
        document.metadata.get(
            "document_name",
            "Unknown"
        )
    )

if len(documents_found) >= 2:

    print("PASS: Multiple documents retrieved.")

    print("\nDocuments found:")

    for document_name in documents_found:
        print(f"- {document_name}")

else:

    print("FAIL: Multiple documents were not retrieved.")


# ==========================================
# TEST 3 — OUT OF CONTEXT
# ==========================================

print("\n================================")
print("TEST 3 — OUT OF CONTEXT")
print("================================")

question = "What is the population of Japan in 2026?"

results, accepted = retrieve(question)

if not accepted:

    print("PASS: Out-of-context question rejected.")

else:

    print("FAIL: Out-of-context information was retrieved.")

    for document, score in accepted:
        print(
            f"- Score: {score:.4f} | "
            f"{document.metadata.get('document_name')}"
        )


# ==========================================
# TEST 4 — SOURCE METADATA
# ==========================================

print("\n================================")
print("TEST 4 — SOURCE METADATA")
print("================================")

question = "What are the four dimensions of food security?"

results, accepted = retrieve(question)

metadata_valid = True

for document, score in accepted:

    metadata = document.metadata

    required_fields = [
        "document_name",
        "organization",
        "domain",
        "publication_year"
    ]

    for field in required_fields:

        if not metadata.get(field):
            metadata_valid = False
            print(f"Missing metadata: {field}")


if metadata_valid:

    print("PASS: Source metadata is available.")

else:

    print("FAIL: Source metadata is incomplete.")


# ==========================================
# FINAL RESULT
# ==========================================

print("\n================================")
print("       EVALUATION COMPLETE")
print("================================")

print("\nRAG pipeline checks completed:")
print("1. Known-answer retrieval")
print("2. Cross-document retrieval")
print("3. Out-of-context rejection")
print("4. Source metadata validation")

print("\nAETHRIX RAG evaluation finished.")