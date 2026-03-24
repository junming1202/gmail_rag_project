from pydantic import BaseModel, Field, conint
from typing import Literal
from ingest import RankOrder, Result
import ollama
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import json

# setup variable
RETRIEVAL_K = 10
DB_PATH = "../preprocessed_db"
collection_name = "docs"
llm_model = "gemma3:1b"

class RankOrder(BaseModel):
    order: list[conint(ge=1, le=RETRIEVAL_K)]

def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant emails from a query of a knowledge base.
The emails are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided emails by relevance to the question, with the most relevant email first.
Reply only with the list of ranked emails ids, nothing else. Include all the emails ids you are provided with, reranked.
Ensure you use each index exactly once.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the emails by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the emails:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# EMAIL ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked email ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_attempt = 5
    attempt = 0
    while attempt < max_attempt:
        try:
            attempt += 1
            response = ollama.chat(model=llm_model, 
                                    messages=messages, 
                                    format=RankOrder.model_json_schema())
            reply = response["message"]["content"]
            order = RankOrder.model_validate(json.loads(reply)).order
            if order:
                break
            
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempt:
                print("Max retries reached. Raising exception.")
                raise(last_exception)

    return [chunks[i - 1] for i in order]

# fetch context
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_client = SentenceTransformer(embedding_model)

def fetch_context_unranked(question):
    query = embedding_client.encode(question)

    chroma = PersistentClient(path = DB_PATH)
    collection = chroma.get_or_create_collection(collection_name)
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks

def fetch_context(question):
    chunks = fetch_context_unranked(question)
    return rerank(question, chunks)


SYSTEM_PROMPT = """
You are a knowledgeable, friendly gmail assistant.
You are chatting with a user about their gmail context.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(f"Extract from {chunk.metadata['category']} category:\n{chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the user gmail information.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = ollama.chat(model=llm_model, messages=[{"role": "system", "content": message}])
    return response["message"]["content"]

def normalize_message_content(content):
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        )
    return content

def answer_question(question: str, history: list[dict] = []) -> [str]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """

    history = [{"role": h["role"], "content": normalize_message_content(h["content"])} for h in history]
    print(history)
    query = rewrite_query(question, history)
    chunks = fetch_context(query)
    messages = make_rag_messages(question, history, chunks)
    response = ollama.chat(model=llm_model, messages=messages)
    return response["message"]["content"]
