from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pydantic import BaseModel, Field, model_validator, conint
from typing import Literal
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
import os
import ollama
import json

# setup variable
load_dotenv(find_dotenv(), override = True)
ollama_api_key = os.getenv("OLLAMA_API_KEY")
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)
llm_model = "gemma3:1b"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "../preprocessed_db"
collection_name = "docs"

# Define pydantic object class
class Result(BaseModel):
    page_content: str = Field(description = "email sender, date received and summary")
    metadata: dict

class Email(BaseModel):
    title: str = Field(description = "Email title")
    sender: str = Field(description = "email sender details")
    date_received: str = Field(description = "Date of email received")
    body: str = Field(description = "Email body")
    category: Literal["job", "transaction", "reminder", "project", "other"]
    summary: str = Field(description = "Email summary based on title and body")

    def as_result(self):
        metadata = {"sender": self.sender, "date_received": self.date_received, "category" : self.category}
        return Result(page_content = self.sender + " send an email on " + self.date_received + ".\nEmail summary: " + self.summary, metadata = metadata)

class Emails(BaseModel):
    emails: list[Email]

class Email_llm_output(BaseModel):
    category: Literal["job", "transaction", "reminder", "project", "other"]
    summary: str = Field(description = "Email summary based on title and body")

# define system_prompt
system_prompt = f"""
You will be given the title and body of an email, some emails might not have any plain text for its body.
Based on the information, respond in json format only that contains category and a summary with the structure below.
The category can only strictly be one of the five categories ["job", "transaction", "reminder", "project", "other"].
No other category is allowed.
For example,
{{
  "category": "transaction"
  "summary": string that contains summary of email title and body
}}
"""


def preprocess_emails(emails):
    for email in tqdm(emails):
        title = email["title"]
        body = email["body"]
        user_prompt = f""" 
        Please help to provide category and summary of the email based on title and/or body. 
        Keep the summary to one or several sentences that are clear, concise and straight to the point. 

        Here is the title of the email:
        {title}


        {"Here is the body of the email: " + body if body else "This email body does not contain any text. Please categorize and summarize based on the title only."}
        """
        messages = [{"role":"system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

        # response = llm_client.chat.completions.create(model = llm_model, messages = messages, max_tokens = 2000)
        attempt = 0
        max_attempt = 5
        while attempt < max_attempt:
            attempt += 1
            try:
                response = ollama.chat(model = llm_model, 
                                    messages = messages, 
                                    format=Email_llm_output.model_json_schema(),
                                    options={
                                            'num_predict': 2000,  # This is Ollama's version of max_tokens
                                        })
                raw_content = response["message"]["content"]
                data = Email_llm_output.model_validate(json.loads(raw_content))
                if data:
                    break

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_attempt:
                    print("Max retries reached. Raising exception.")
                    raise(last_exception)

        email["category"] = data.category
        email["summary"] = data.summary

    emails = Emails.model_validate({"emails":emails})
    return emails


# setup embedding model
embedding_client = SentenceTransformer(embedding_model)

def create_embeddings(emails):
    chroma = PersistentClient(path = DB_PATH)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [email.as_result().page_content for email in emails.emails]

    vectors = embedding_client.encode(texts)

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(emails.emails))]
    metas = [email.as_result().metadata for email in emails.emails]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")