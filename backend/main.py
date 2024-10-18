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
                "You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements, budget, and room specifications, generate a detailed sales quotation that includes three tiers: Basic, Premium, and Ultimate.<|eot_id|>\n\n"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{input_text}<|eot_id|>\n\n"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "parameters": {
                "max_new_tokens": 3096,  # Increased token limit for larger outputs
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
You are a state-of-the-art sales assistant for Speaker Selling. Based on the customer's requirements, budget, and room specifications, generate a detailed sales quotation that includes three tiers: Basic, Premium, and Ultimate.

**Customer Details**:
- **Name**: {client_name}
- **Email**: {email}
- **Address**: {client_address}
- **Type of Build**: {type_of_build}
- **Requirements**: {requirements}
- **Budget**: {budget}

**Available Products**:
{available_products}

**Budget Tiers**:

- **Basic**:
  - Recommend essential products that meet the minimum requirements while strictly staying within the budget.
  - Ensure the product type matches the room requirements, especially for wall-mounted speakers or specific configurations.
  - Carefully review the long descriptions to identify products with the necessary features.

- **Premium**:
  - Provide cost-effective options that enhance quality and functionality.
  - Consider higher-rated (rating) products and balance cost with budget.
  - Calculate and suggest the correct quantity of speakers needed for each room based on room size and acoustic needs.
  - Ensure the total cost fits within the budget.
  - Thoroughly check long descriptions and product category, sub-category and type of product to ensure feature requirements.

- **Ultimate**:
  - Offer the best available products that maximize quality and features.
  - Include top-rated (High Rating) items optimal setup.
  - Calculate and suggest the correct quantity of speakers needed for each room based on room size and acoustic needs.
  - It's acceptable if the total cost slightly exceeds the budget (Budget can exceed by 20-30%) to provide significant value.
  - Carefully match product types and ratings to the customer's requirements, especially for specialized needs.
  - Thoroughly check long descriptions and product category, sub-category and type to include products with advanced features.

**Instructions**:

1. **Pricing**: Use only the unit prices provided in the available products list. Do not assume or estimate prices.
2. **Product Selection**:
   - For each room, recommend the most appropriate product(s) from the catalog.
   - Ensure the quantity of speakers matches the room size and customer's usage requirements. If no quantity of speaker is provided assume best scenario of number of speaker required for that room.
   - Pay special attention to the type (e.g., floor, wall, ceiling, hidden) of speakers from product catalog and ratings provided in product catalog of the speakers to meet specific room needs.
   - Review the long descriptions to verify that the products have the required features and it meets the product type requirements for room.
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
     - Reason (Provide reasoning for selection)
4. **Out of Budget**:
   - If no available product fits a room's requirement within the budget, list "Not Recommended (Out of Budget)" for that room in the respective budget tier.
5. **Output Format**:
   - Output the final result in the exact structured JSON format provided in section **Output JSON Structure** for all three budget tiers.
   - Ensure the JSON is valid and can be used directly for processing without any extra additional text or explanations other then JSON requested.

**Output JSON Structure**:
{{
  "client_name": "Customer Name",
  "client_email": "Customer Email",
  "client_address": "Customer Address",
  "type_of_build": "Customer's requested build (e.g., New Build, Condo, Retrofit)",
  "budgets": [
    {{
      "type": "Basic",
      "rooms": {{
        "Room Name": {{
          "requirement": "Requirement (e.g., Wall, Ceiling, Floor, Hidden or Number of Speakers type)",
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
        // Repeat for each room
      }}
    }},
    // Repeat for "Premium" and "Ultimate" budget tiers
  ]
}}
** NOTE : MAKE SURE NO TEXT IS RESPONDED UNDER JSON OUTPUT**
"""
# Step 10: Create a function to process queries

response_cleaning_template = """
You are an assistant that only outputs valid JSON without any additional text.
Extract the following information from the text and output it as valid formatted JSON:

{response}
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
    relevant_keywords = json.dumps(query)
    room_requirements = "\n".join(
        [f"{products}" for room, products in query.items() if room not in ["client_name", "client_address", "type_of_build", "budget", "email", "timestamp"] and products is not None])

    room_requirements_ = "\n".join(
        [f"- In **{room}**: Following type of speaker are needed to be installed : {products}" for room, products in query.items() if room not in ["client_name", "client_address", "type_of_build", "budget", "email", "timestamp"] and products is not None])

    # Retrieve relevant documents from the vector store
    relevant_docs = vector_store.similarity_search(room_requirements, k=5)

    # Format the retrieved documents into a string
    available_products = '\n'.join([doc.page_content for doc in relevant_docs])

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

    # Invoke SageMaker model for the response
    answer = invoke_sagemaker_model(prompt_input)
    # print(f"Found initial response: {answer}")

    # if answer:
    #     # Clean and parse the response
    #     prompt_cleaning_input = response_cleaning_template.format(
    #         response=answer)
    #     cleaned_response = invoke_sagemaker_model(prompt_cleaning_input)

    #     if cleaned_response:
    json_response = json.loads(answer)
    print("FOUND JSON", json_response)
    # Load JSON data
    data = json_response

    # Setup Jinja2 environment
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    # Load the template
    template = env.get_template('template.html')

    # Render the template with dynamic data
    output = template.render(client_name=data['client_name'],
                             client_email=data['client_email'],
                             client_address=data['client_address'],
                             type_of_build=data['type_of_build'],
                             budgets=data['budgets'])

    send_html_email(
        query['email'], "Sales Quote for Home Automation", output)
    return json_response


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
