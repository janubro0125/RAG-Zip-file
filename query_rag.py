import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Relevance scores must be between 0 and 1"
)

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from query_logger import log_query


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

CHROMA_DIR = "data/chroma"
SCORE_THRESHOLD = 0.40


# ==========================================
# LOAD LOCAL EMBEDDINGS
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
# AETHRIX RAG ASSISTANT
# ==========================================

print("\n================================")
print("       AETHRIX RAG ASSISTANT")
print("================================")
print("Ask questions about the knowledge base.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")


    # ==========================================
    # EXIT
    # ==========================================

    if question.lower().strip() == "exit":

        print("\nAETHRIX RAG stopped.")

        break


    # ==========================================
    # EMPTY QUESTION
    # ==========================================

    if not question.strip():

        continue


    print("\nSearching relevant information...")


    # ==========================================
    # RETRIEVE DOCUMENTS
    # ==========================================

    results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=4
    )


    # ==========================================
    # CUSTOM SCORE VALIDATION
    # ==========================================

    documents = []

    for document, score in results:

        if score >= SCORE_THRESHOLD:

            documents.append(
                (document, score)
            )


    # ==========================================
    # REJECT IRRELEVANT QUESTIONS
    # ==========================================

    if not documents:

        log_query(
            question=question,
            status="REJECTED",
            documents_retrieved=0,
            sources="",
            relevance_scores=""
        )

        print(
            "\nI could not find the answer "
            "in the provided documents.\n"
        )

        continue


    # ==========================================
    # BUILD SOURCE-AWARE CONTEXT
    # ==========================================

    context_parts = []

    for document, score in documents:

        metadata = document.metadata

        source = metadata.get(
            "document_name",
            "Unknown"
        )

        year = metadata.get(
            "publication_year",
            "Unknown"
        )

        page = metadata.get("page")

        if page is not None:

            page_number = page + 1

        else:

            page_number = "Unknown"


        context_parts.append(

            f"[Source: {source} | "
            f"Year: {year} | "
            f"Page: {page_number} | "
            f"Relevance: {score:.4f}]\n"
            f"{document.page_content}"

        )


    context = "\n\n".join(context_parts)


    # ==========================================
    # LOG SUCCESSFUL RETRIEVAL
    # ==========================================

    source_names = []

    relevance_scores = []

    for document, score in documents:

        source_names.append(
            document.metadata.get(
                "document_name",
                "Unknown"
            )
        )

        relevance_scores.append(
            f"{score:.4f}"
        )


    log_query(
        question=question,
        status="SUCCESS",
        documents_retrieved=len(documents),
        sources="; ".join(source_names),
        relevance_scores="; ".join(relevance_scores)
    )


    # ==========================================
    # STRICT GROUNDED RAG PROMPT
    # ==========================================

    prompt = f"""
You are AETHRIX, a strict Retrieval-Augmented Generation assistant.

IMPORTANT RULES:

1. Use ONLY the information contained in the retrieved context below.

2. DO NOT use your own knowledge or training knowledge.

3. DO NOT use information from the internet, web search, external
   sources, or any source outside the retrieved context.

4. DO NOT make assumptions or invent information.

5. If the answer cannot be found in the retrieved context, respond
   EXACTLY with:

I could not find the answer in the provided documents.

6. Keep the answer clear, factual, and concise.

7. Every factual statement in your answer must be supported by
   the retrieved context.

8. Prefer the most relevant retrieved passages.

9. Do not introduce facts that are not present in the context.

10. Do not use general world knowledge to complete missing information.

Retrieved Context:

{context}

User Question:

{question}

Answer using ONLY the retrieved context.
"""


    # ==========================================
    # GENERATE ANSWER
    # ==========================================

    print("Generating answer...\n")

    response = llm.invoke(prompt)


    # ==========================================
    # EXTRACT RESPONSE
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

    print("================================")
    print("AETHRIX ANSWER")
    print("================================")

    print(
        answer.strip()
    )


    # ==========================================
    # DISPLAY SOURCES
    # ==========================================

    print("\n================================")
    print("SOURCES")
    print("================================")

    seen_pages = set()

    for document, score in documents:

        metadata = document.metadata

        source = metadata.get(
            "document_name",
            "Unknown"
        )

        year = metadata.get(
            "publication_year",
            "Unknown"
        )

        page = metadata.get("page")

        if page is not None:

            page_number = page + 1

        else:

            page_number = "Unknown"


        source_key = (
            source,
            year,
            page_number
        )


        if source_key not in seen_pages:

            print(
                f"- {source} | "
                f"Year: {year} | "
                f"Page: {page_number} | "
                f"Relevance: {score:.4f}"
            )

            seen_pages.add(
                source_key
            )


    print()