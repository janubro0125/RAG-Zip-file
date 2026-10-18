import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# CONFIGURATION
# ==========================================

PDF_DIR = "data/pdfs"
CHROMA_DIR = "data/chroma"

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 300


# ==========================================
# CHECK PDF DIRECTORY
# ==========================================

os.makedirs(PDF_DIR, exist_ok=True)

print("================================")
print("     AETHRIX DOCUMENT INGESTION")
print("================================")


# ==========================================
# FIND PDF FILES
# ==========================================

pdf_files = [
    file
    for file in os.listdir(PDF_DIR)
    if file.lower().endswith(".pdf")
]


if not pdf_files:

    print("\nNo PDF files found in:")
    print(PDF_DIR)

    raise SystemExit


print(f"\nPDF files found: {len(pdf_files)}")

for file in pdf_files:

    print(f"- {file}")


# ==========================================
# SELECT DOCUMENT
# ==========================================

if len(pdf_files) == 1:

    selected_pdf = pdf_files[0]

else:

    print("\nEnter the PDF filename to ingest.")

    selected_pdf = input(
        "\nPDF filename: "
    ).strip()


if selected_pdf not in pdf_files:

    print("\nERROR: PDF file not found.")

    raise SystemExit


pdf_path = os.path.join(
    PDF_DIR,
    selected_pdf
)


# ==========================================
# VALIDATE PDF
# ==========================================

print("\nValidating PDF...")

with open(
    pdf_path,
    "rb"
) as file:

    header = file.read(4)


if header != b"%PDF":

    print("ERROR: Invalid PDF file.")

    raise SystemExit


print("PDF validation: PASS")


# ==========================================
# EXTRACT DOCUMENT INFORMATION
# ==========================================

filename = os.path.basename(
    pdf_path
)

name_without_extension = os.path.splitext(
    filename
)[0]


# ==========================================
# ORGANIZATION
# ==========================================

organization = name_without_extension.split("_")[0]

if not organization:

    organization = "Unknown"


# ==========================================
# PUBLICATION YEAR
# ==========================================

year_match = re.search(
    r"(19|20)\d{2}",
    filename
)


if year_match:

    publication_year = int(
        year_match.group(0)
    )

else:

    publication_year = "Unknown"


# ==========================================
# LOAD PDF
# ==========================================

print("\nLoading PDF...")

loader = PyPDFLoader(
    pdf_path
)

documents = loader.load()


if not documents:

    print("ERROR: No pages extracted.")

    raise SystemExit


print(
    f"PDF pages loaded: {len(documents)}"
)


# ==========================================
# ADD METADATA
# ==========================================

print("\nAdding metadata...")

for document in documents:

    document.metadata["organization"] = (
        organization
    )

    document.metadata["domain"] = (
        "Knowledge Base"
    )

    document.metadata["publication_year"] = (
        publication_year
    )

    document.metadata["document_name"] = (
        filename
    )


# ==========================================
# SPLIT DOCUMENT
# ==========================================

print("\nSplitting document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(
    documents
)


if not chunks:

    print("ERROR: No chunks created.")

    raise SystemExit


print(
    f"Text chunks created: {len(chunks)}"
)


# ==========================================
# CREATE LOCAL EMBEDDINGS
# ==========================================

print("\nLoading local embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# LOAD CHROMADB
# ==========================================

print("\nLoading ChromaDB...")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)


# ==========================================
# REMOVE EXISTING VERSION
# ==========================================

print("\nChecking for existing document...")

existing = vectorstore.get(
    where={
        "document_name": filename
    }
)


existing_ids = existing.get(
    "ids",
    []
)


if existing_ids:

    print(
        f"Existing chunks found: "
        f"{len(existing_ids)}"
    )

    print("Removing old version...")

    vectorstore.delete(
        ids=existing_ids
    )

    print("Old version removed.")

else:

    print("No existing version found.")


# ==========================================
# CREATE UNIQUE CHUNK IDs
# ==========================================

chunk_ids = []

for index, document in enumerate(
    chunks
):

    page = document.metadata.get(
        "page",
        0
    )

    chunk_id = (
        f"{filename}_"
        f"page_{page}_"
        f"chunk_{index}"
    )

    chunk_ids.append(
        chunk_id
    )


# ==========================================
# STORE DOCUMENT
# ==========================================

print("\nCreating embeddings and storing...")

vectorstore.add_documents(
    documents=chunks,
    ids=chunk_ids
)


# ==========================================
# SUCCESS
# ==========================================

print("\n================================")
print("       INGESTION SUCCESS")
print("================================")

print(
    f"Document: {filename}"
)

print(
    f"Pages: {len(documents)}"
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    f"Organization: {organization}"
)

print(
    f"Publication Year: "
    f"{publication_year}"
)

print(
    f"ChromaDB: {CHROMA_DIR}"
)

print("\nDocument ingestion completed successfully.")