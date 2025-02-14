from uuid import uuid4

import chromadb
from chromadb.config import Settings

from tqdm import tqdm


settings = Settings(allow_reset=True)
client = chromadb.PersistentClient(path="./chromadb", settings=settings)
collection = client.get_or_create_collection("documents")


def add_documents_(documents, ids=None, metadatas=None):
    collection = client.get_or_create_collection("documents")
    if not ids:
        ids = [str(uuid4()) for _ in range(len(documents))]
    if not metadatas:
        metadatas = [None for _ in range(len(documents))]
    if not documents:
        return False
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return True


def add_pdf_documents_(
    chapter_pages, chapter_page_numbers, chapter_titles, document_title=""
):
    collection = client.get_or_create_collection("pdf_documents")
    for pages, page_numbers, chapter_title in tqdm(
        zip(chapter_pages, chapter_page_numbers, chapter_titles),
        total=len(chapter_pages),
    ):
        full_title = "/".join(chapter_title)
        for page, page_number in zip(pages, page_numbers):
            collection.add(
                documents=[page],
                metadatas=[{"title": full_title, "page": page_number}],
                ids=[f"{document_title}_{page_number}"],
            )
    return True


def get_documents_():
    collection = client.get_or_create_collection("documents")
    documents = collection.get()
    return documents


def get_pdf_documents_():
    collection = client.get_or_create_collection("pdf_documents")
    documents = collection.get()
    return documents


def query_documents_(query_texts):
    collection = client.get_or_create_collection("documents")
    return collection.query(query_texts=query_texts, n_results=2)


def query_pdf_documents_(query_texts):
    collection = client.get_or_create_collection("pdf_documents")
    return collection.query(query_texts=query_texts, n_results=2)
