# main.py

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from fastapi import FastAPI, HTTPException, BackgroundTasks
from core.config import settings
import os
import sqlite3
import re
import numpy as np
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from jinja2 import Environment, FileSystemLoader
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from openai import OpenAI
import time


load_dotenv()

gptClient = OpenAI()


# Define the function to invoke SageMaker model


def invoke_sagemaker_model(input_text):
    try:
        sagemaker_client = boto3.client(
            'sagemaker-runtime', region_name="us-east-1")

# Define the SageMaker endpoint
        SAGEMAKER_ENDPOINT = "llama-3-1-8b"

        # Prepare the payload for SageMaker
        payload = json.dumps({
            "inputs": (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements, budget, and room specifications, generate a detailed sales quotation that includes three tiers: Good, Better, and Best.<|eot_id|>\n\n"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{input_text}<|eot_id|>\n\n"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "parameters": {
                "max_new_tokens": 30096,  # Increased token limit for larger outputs
                "top_p": 0.9,
                "temperature": 0.6
            }

        })

        # Call the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=payload
        )

        # Extract response
        result = json.loads(response['Body'].read().decode())
        # Modify this based on your model's response format
        return result['generated_text']
    except Exception as e:
        print(f"Error invoking SageMaker: {e}")
        return None


class InferenceRequest(BaseModel):

    client_name: Optional[str] = None
    client_address: Optional[str] = None
    budget: Optional[str] = None
    email: Optional[str] = None
    type_of_build: Optional[str] = None
    hallway: Optional[str] = None
    lounge: Optional[str] = None
    dining_room: Optional[str] = None
    kitchen: Optional[str] = None
    bedroom_1: Optional[str] = None
    bedroom_2: Optional[str] = None
    bedroom_3: Optional[str] = None
    terrace: Optional[str] = None
    pool_area: Optional[str] = None


# Step 1: Set up ChromaDB persistent directory
PERSIST_DIRECTORY = "db"
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Load the Product Catalog File Here

loader = CSVLoader(file_path="./product_catalog.csv")

data = loader.load()


# Step 3: Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Reduced chunk size
    chunk_overlap=50
)
docs = text_splitter.split_documents(data)


# Step 4: Create embeddings using a local model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)


# Step 5: Create or load a vector store for embeddings
collection_name = "product_catalogs2"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    client=client,
    persist_directory=PERSIST_DIRECTORY
)

# Step 6: Check if the collection is empty; if so, add documents

batch_size = 500  # Adjust this based on your model's batch size limits

# Check if the vector store collection exists and contains data
# (Assuming the collection's count method returns the number of documents)
if vector_store._collection.count() == 0:  # Check if the vector store is empty
    print("Vector store is empty, adding documents.")
    # Split the documents into smaller batches
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        vector_store.add_documents(batch_docs)

    try:

        vector_store.persist()  # Save the new vector store
    except:
        print("Deprecated API Call found")
    print("Created new vector store and added documents in batches.")
else:
    print("Vector store already contains data, skipping document addition.")
# Step 7: Set up the LLM (Ollama)
# llm = OllamaLLM(model="llama3")


prompt_template = """
You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements, budget, and room specifications, generate a detailed sales quotation that includes three tiers: Good, Better, and Best.

**Customer Details**:
- **Name**: {client_name}
- **Email**: {email}
- **Address**: {client_address}
- **Type of Build**: {type_of_build}
- **Requirements**: {requirements}
- **Minimum Budget**: {budget}

**Available Products**:
{available_products}

**Budget Tiers**:

- **Good**:
  - Choose products with a **Rating of 4**. Only products that have a rating of 4 should be in this tier.
  - Ensure that the selected products match the **Type**, **Subcategory**.
  - Exceed the budget if necessary to meet the requirements.
  
- **Better**:
  - Provide high-quality products that enhance functionality and coverage, staying close to the budget.
  - Ensure that the products match the **Type**, **Subcategory**.
  - Exceed the budget if necessary to meet the requirements.

- **Best**:
  - Choose products with a **Rating of 6**. Only products that have a rating of 6 should be in this tier.
  - Ensure that products in this tier match the **Type**, **Subcategory**
  - Exceed the budget if necessary to meet the requirements.

**Instructions**:

1. **Pricing**: Use only the unit prices provided in the available products list.
2. **Product Selection**:
   - For each room, recommend products that matches the **Type**, **Subcategory**.
   - Prioritize matching the **Type** and **Subcategory** fields
3. **Product Details**:
   - Include comprehensive information for each product:
     - Name, Part Number, Category, Subcategory, Type, Rating, Short Description, Long Description, Quantity, Unit Price, Reason for selection.
4. **Output Format**:
   - The final output should be in valid JSON format without additional text or explanations.

**Output JSON Structure**:
{{
  "client_name": "Customer Name",
  "client_email": "Customer Email",
  "client_address": "Customer Address",
  "type_of_build": "Customer's requested build (e.g., New Build, Condo, Retrofit)",
  "budgets": [
    {{
      "type": "Good",
      "rooms": {{
        "Room Name": {{
          "requirement": "Requirement",
          "products": [
            {{
              "name": "Product Name",
              "part_number": "Part Number",
              "category": "Category",
              "subcategory": "Subcategory",
              "type": "Type",
              "rating": "Rating",
              "short_description": "Short Description",
              "long_description": "Long Description",
              "quantity": "Quantity",
              "unit_price": "Unit Price",
              "reason": "Reason for selection if not an exact match"
            }}
          ]
        }}
      }}
    }},
    {{... for "Better" and "Best" budget tiers}}
  ]
}}
** NOTE : FINAL OUTPUT SHOULD BE ONLY VALID JSON**
"""

# Step 10: Create a function to process queries

response_cleaning_template = """
You are an assistant that only outputs valid JSON without any additional text.
Extract the following information from the text and output it as valid formatted JSON:

{response}
"""


def send_html_email(recipient, subject, body_html, aws_region="eu-north-1"):
    ses_client = boto3.client('ses', region_name=aws_region)
    try:
        response = ses_client.send_email(
            Source=os.getenv('SES_EMAIL'),
            Destination={'ToAddresses': [recipient]},
            Message={'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                     'Body': {'Html': {'Data': body_html, 'Charset': 'UTF-8'}}}
        )
        print("Email sent! Message ID:", response['MessageId'])
    except NoCredentialsError:
        print("Error: No AWS credentials found.")
    except PartialCredentialsError:
        print("Error: Partial credentials found.")
    except Exception as e:
        print(f"Error: {e}")


# llm = OllamaLLM(model="llama3")


def process_query(query, budget):
    for attempt in range(3):  # Retry up to 3 times
        try:
            relevant_keywords = json.dumps(query)
            room_requirements = "\n".join(
                [f"{products}" for room, products in query.items() if room not in [
                    "client_name", "client_address", "type_of_build", "budget", "email", "timestamp"] and products is not None]
            )

            room_requirements_ = "\n".join(
                [f"- In **{room}**: Following type of speaker are needed to be installed : {products}" for room, products in query.items(
                ) if room not in ["client_name", "client_address", "type_of_build", "budget", "email", "timestamp"] and products is not None]
            )

            # Retrieve relevant documents from the vector store
            relevant_docs = vector_store.similarity_search(
                room_requirements, k=5)

            # Format the retrieved documents into a string
            available_products = '\n'.join(
                [doc.page_content for doc in relevant_docs])

            # Prepare the input text for SageMaker
            prompt_input = prompt_template.format(
                client_name=query["client_name"],
                client_address=query["client_address"],
                email=query["email"],
                type_of_build=query["type_of_build"],
                requirements=room_requirements_,
                budget=budget,
                available_products=available_products
            )

            completion = gptClient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements, budget, and room specifications, generate a detailed sales quotation that includes three tiers: Good, Better, and Best. Only return a valid JSON output."},
                    {"role": "user", "content": prompt_input},
                    {"role": "assistant", "content": "Respond only with valid JSON data. Do not include any extra text, explanations, or formatting like triple backticks or any markdown."}
                ]
            )

            answer = completion.choices[0].message.content

            # Convert the response to JSON
            json_response = json.loads(answer)

            # Setup Jinja2 environment
            file_loader = FileSystemLoader('.')
            env = Environment(loader=file_loader)

            # Load the template
            template = env.get_template('template.html')

            # Render the template with dynamic data
            output = template.render(client_name=json_response['client_name'],
                                     client_email=json_response['client_email'],
                                     client_address=json_response['client_address'],
                                     type_of_build=json_response['type_of_build'],
                                     budgets=json_response['budgets'],
                                     budget=budget
                                     )

            # Send email with generated HTML content
            send_html_email(
                query['email'], "Sales Quote for Home Automation", output
            )

            return json_response

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < 2:  # Wait before retrying, if not on the last attempt
                time.sleep(4)
            else:
                print("All attempts failed.")
                raise  # Raise the exception if all attempts fail


app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)


@ app.get("/")
def hello_api():
    return {"msg": "Hello Associates🚀"}


@ app.post("/inference/")
def inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    try:

        background_tasks.add_task(
            process_query, request.dict(), request.budget)

        # Call the process_query function to get the HTML response
        # result = process_query(
        #     query=request.requirements,
        #     budget=request.budget,
        #     chat_history=request.chat_history
        # )

        # Convert the inference result (HTML) to PDF
        # pdf_file_path = html_to_pdf(result)

        return {"message": "ok"}
    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=500, detail=f"Error during inference: {str(e)}")
