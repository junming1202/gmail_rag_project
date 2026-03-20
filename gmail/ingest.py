from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

DB_PATH = "../preprocessed_db"
collection_name = "docs"


def create_embeddings(emails):
    

    chroma = PersistentClient(path = DB_PATH)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_client = SentenceTransformer(embedding_model)

    texts = [email.as_result().page_content for email in emails.emails]

    vectors = embedding_client.encode(texts)

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(emails.emails))]
    metas = [email.as_result().metadata for email in emails.emails]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")