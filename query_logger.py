import csv
import os
from datetime import datetime


LOG_DIR = "data/logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "rag_queries.csv"
)


# ==========================================
# CREATE LOG DIRECTORY
# ==========================================

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ==========================================
# INITIALIZE CSV FILE
# ==========================================

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "question",
            "status",
            "documents_retrieved",
            "sources",
            "relevance_scores"
        ])


# ==========================================
# LOG RAG QUERY
# ==========================================

def log_query(
    question,
    status,
    documents_retrieved=0,
    sources="",
    relevance_scores=""
):

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            question,
            status,
            documents_retrieved,
            sources,
            relevance_scores
        ])


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    log_query(
        question="Test query",
        status="TEST",
        documents_retrieved=1,
        sources="FAO_test.pdf",
        relevance_scores="0.75"
    )

    print("Query logger test completed.")
    print(f"Log file: {LOG_FILE}")