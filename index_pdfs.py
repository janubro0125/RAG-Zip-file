import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PDF_DIR = "data/pdfs"
CHROMA_DIR = "data/chroma"


print("Loading PDFs...")

loader = PyPDFDirectoryLoader(PDF_DIR)
documents = loader.load()

print(f"PDF pages loaded: {len(documents)}")

if not documents:
    raise ValueError("No PDF documents found in data/pdfs")


print("Adding document metadata...")


for document in documents:

    source = document.metadata.get("source", "")

    filename = os.path.basename(source)

    document.metadata["organization"] = "FAO"
    document.metadata["domain"] = "Food and Agriculture"

    if "2023" in filename:
        document.metadata["publication_year"] = 2023
    else:
        document.metadata["publication_year"] = 2013

    document.metadata["document_name"] = filename


print("Splitting documents...")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)

print(f"Text chunks created: {len(chunks)}")


print("Creating local Hugging Face embeddings...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Storing embeddings in ChromaDB...")


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)


print("\nSUCCESS!")
print(f"Indexed chunks: {len(chunks)}")
print(f"ChromaDB location: {CHROMA_DIR}")
print("Metadata added: organization, domain, publication_year, document_name")
