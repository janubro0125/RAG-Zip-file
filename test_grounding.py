import warnings

warnings.filterwarnings(
    "ignore",
    message="Relevance scores must be between 0 and 1"
)

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

CHROMA_DIR = "data/chroma"
SCORE_THRESHOLD = 0.40


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

print("Loading local embeddings...")

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
# LOAD GEMINI
# ==========================================

print("Loading Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# ==========================================
# TEST QUESTION
# ==========================================

question = "What are the four dimensions of food security?"


print("\n================================")
print("       GROUNDING EVALUATION")
print("================================")

print("\nQuestion:")
print(question)


# ==========================================
# RETRIEVE DOCUMENTS
# ==========================================

print("\nRetrieving relevant documents...")

results = vectorstore.similarity_search_with_relevance_scores(
    question,
    k=4
)


documents = []

for document, score in results:

    if score >= SCORE_THRESHOLD:

        documents.append(
            (document, score)
        )


# ==========================================
# CHECK RETRIEVAL
# ==========================================

if not documents:

    print("\nFAIL: No relevant documents retrieved.")

    raise SystemExit


print(
    f"\nRetrieved documents: {len(documents)}"
)


# ==========================================
# BUILD CONTEXT
# ==========================================

context_parts = []

for document, score in documents:

    metadata = document.metadata

    source = metadata.get(
        "document_name",
        "Unknown"
    )

    page = metadata.get(
        "page",
        "Unknown"
    )

    if page != "Unknown":

        page = page + 1


    context_parts.append(

        f"[Source: {source} | "
        f"Page: {page} | "
        f"Relevance: {score:.4f}]\n"
        f"{document.page_content}"

    )


context = "\n\n".join(context_parts)


# ==========================================
# GROUNDED GENERATION PROMPT
# ==========================================

prompt = f"""
You are evaluating a strict RAG system.

Use ONLY the retrieved context below.

Do not use outside knowledge.
Do not use internet information.
Do not add facts that are not present in the context.

Retrieved Context:

{context}

Question:

{question}

Provide a concise answer using only the retrieved context.
"""


# ==========================================
# GENERATE ANSWER
# ==========================================

print("\nGenerating grounded answer...")

response = llm.invoke(prompt)


# ==========================================
# EXTRACT ANSWER
# ==========================================

if isinstance(response.content, list):

    answer = ""

    for item in response.content:

        if (
            isinstance(item, dict)
            and item.get("type") == "text"
        ):

            answer += item.get(
                "text",
                ""
            )

else:

    answer = str(
        response.content
    )


# ==========================================
# DISPLAY ANSWER
# ==========================================

print("\n================================")
print("GENERATED ANSWER")
print("================================")

print(answer.strip())


# ==========================================
# DISPLAY CONTEXT SOURCES
# ==========================================

print("\n================================")
print("CONTEXT SOURCES")
print("================================")

for document, score in documents:

    metadata = document.metadata

    source = metadata.get(
        "document_name",
        "Unknown"
    )

    page = metadata.get(
        "page",
        "Unknown"
    )

    if page != "Unknown":

        page = page + 1


    print(
        f"- {source} | "
        f"Page: {page} | "
        f"Relevance: {score:.4f}"
    )


# ==========================================
# GROUNDING TEST
# ==========================================

print("\n================================")
print("GROUNDING TEST")
print("================================")

answer_lower = answer.lower()

expected_terms = [
    "food availability",
    "economic and physical access",
    "food utilization",
    "stability"
]

matched_terms = []

for term in expected_terms:

    if term in answer_lower:

        matched_terms.append(term)


if len(matched_terms) == len(expected_terms):

    print("PASS: Answer contains the expected facts.")

    print(
        f"Matched facts: "
        f"{len(matched_terms)}/{len(expected_terms)}"
    )

else:

    print("WARNING: Some expected facts were not found.")

    print(
        f"Matched facts: "
        f"{len(matched_terms)}/{len(expected_terms)}"
    )


print("\n================================")
print("GROUNDING EVALUATION COMPLETE")
print("================================")