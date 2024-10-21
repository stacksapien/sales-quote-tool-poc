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


# Define the function to invoke SageMaker model


def invoke_sagemaker_model(input_text):
    try:
        sagemaker_client = boto3.client('sagemaker-runtime', region_name="us-east-1", aws_access_key_id=os.getenv(
            'AWS_ACCESS_KEY'), aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))

# Define the SageMaker endpoint
        SAGEMAKER_ENDPOINT = "jumpstart-dft-llama-3-1-8b-instruct-20241017-231713"

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
llm = OllamaLLM(model="llama3")


prompt_template = """
You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements for all rooms and the current budget tier, generate a detailed sales quotation.

**Customer Details**:
- **Name**: {client_name}
- **Email**: {email}
- **Address**: {client_address}
- **Type of Build**: {type_of_build}

**Room Requirements**:
{requirements}

**Budget**: {budget}
**Budget Tier**: {budget_tier}

**Available Products**:
{available_products}

**Instructions**:

1. **Pricing**: Use only the unit prices provided in the available products list.
2. **Product Selection**:
   - Recommend appropriate product(s) for each room.
   - Ensure the quantity matches the room size and customer's usage requirements.
3. **Product Details**:
   - Provide comprehensive information for each product, including:
     - Name
     - Part Number
     - Category
     - Subcategory
     - Type
     - Rating
     - Short Description
     - Long Description
     - Quantity
     - Unit Price
     - Reason
4. **Output Format**:
   - Output the final result in the exact structured JSON format provided below.
   - Ensure the JSON is valid and does not include any additional text.

**Output JSON Structure**:
{{
  "rooms": {{
    "{room_name}": {{
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
          "reason": "Reason for selection"
        }}
      ]
    }}
  }}
}}
**NOTE: ONLY OUTPUT THE JSON RESPONSE.**
"""


def send_html_email(recipient, subject, body_html, aws_region="eu-north-1"):
    ses_client = boto3.client('ses', region_name=aws_region, aws_access_key_id=os.getenv(
        'AWS_ACCESS_KEY'), aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
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


def process_query(query, budget):
    # Extract client details
    client_details = {
        "client_name": query.get("client_name"),
        "client_address": query.get("client_address"),
        "email": query.get("email"),
        "type_of_build": query.get("type_of_build")
    }

    # Initialize the final JSON structure
    final_output = {
        "client_name": client_details["client_name"],
        "client_email": client_details["email"],
        "client_address": client_details["client_address"],
        "type_of_build": client_details["type_of_build"],
        "budgets": []
    }

    # Define budget tiers
    budget_tiers = ["Good", "Better", "Best"]

    for tier in budget_tiers:
        tier_data = {
            "type": tier,
            "rooms": {}
        }

        room_requirements_list = []
        available_products_list = []

        for room, requirement in query.items():
            if room in ["client_name", "client_address", "type_of_build", "budget", "email", "timestamp"] or not requirement:
                continue

            room_requirements = f"{requirement}"
            relevant_docs = vector_store.similarity_search(
                room_requirements, k=3)

            # Format the retrieved documents into a string
            available_products = '\n'.join(
                [doc.page_content for doc in relevant_docs])

            room_requirements_list.append(f"In **{room}**: {requirement}")
            available_products_list.append(available_products)

        # Combine room requirements and available products for the current budget tier
        combined_room_requirements = '\n'.join(room_requirements_list)
        combined_available_products = '\n'.join(available_products_list)

        # Prepare the input text for SageMaker
        prompt_input = prompt_template.format(
            client_name=client_details["client_name"],
            client_address=client_details["client_address"],
            email=client_details["email"],
            type_of_build=client_details["type_of_build"],
            requirements=combined_room_requirements,
            budget=budget,
            room_name=room,
            available_products=combined_available_products,
            budget_tier=tier
        )

        # Invoke SageMaker model for the response
        answer = invoke_sagemaker_model(prompt_input)

        if answer:
            try:
                json_response = json.loads(answer)
                tier_data["rooms"] = json_response["rooms"]
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue  # Implement retry logic here if needed

        final_output["budgets"].append(tier_data)

    # After processing all rooms and tiers, send the email
    # Render the template with dynamic data
    print(json.dumps(final_output))
    output_html = render_html_template(final_output)

    send_html_email(
        client_details['email'], "Sales Quote for Home Automation", output_html
    )
    return final_output


def render_html_template(data):
    # Setup Jinja2 environment
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    # Load the template
    template = env.get_template('template.html')

    # Render the template with dynamic data
    output = template.render(
        client_name=data['client_name'],
        client_email=data['client_email'],
        client_address=data['client_address'],
        type_of_build=data['type_of_build'],
        budgets=data['budgets']
    )
    return output


load_dotenv()
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
