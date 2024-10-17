from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from fastapi import FastAPI, HTTPException, BackgroundTasks
from core.config import settings
import os
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


class InferenceRequest(BaseModel):
    budget: Optional[str] = None
    email: Optional[str] = None
    dining: Optional[str] = None
    living: Optional[str] = None
    library: Optional[str] = None
    entry: Optional[str] = None
    breakfast_room: Optional[str] = None
    office: Optional[str] = None
    kitchen: Optional[str] = None
    primary_bedroom: Optional[str] = None
    primary_closet: Optional[str] = None
    primary_bathroom: Optional[str] = None
    bedroom_1: Optional[str] = None
    bedroom_2: Optional[str] = None
    bedroom_3: Optional[str] = None
    guest_room: Optional[str] = None
    timestamp: Optional[str] = None


# Step 1: Load the product catalog
file_path = './product_catalog.csv'  # Change this to the actual path
product_catalog = pd.read_csv(file_path)

# Extract only the columns we care about
product_catalog = product_catalog[['Manufacturer', 'Model', 'Category',
                                   'Subcategory', 'Part Number', 'Short Description', 'Unit Price']]

# Step 2: Extract keywords from the room requirements
# Step 2: Initialize HuggingFace Embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)


# Step 3: Extract keywords from the room requirements
def extract_product_keywords(query):
    keywords = []
    for room, products in query.items():
        if room not in ["budget", "email", "timestamp"]:
            keywords.extend([product.strip()
                            for product in products.split(",")])
    return keywords

# Step 4: Similarity search using embeddings and cosine similarity


# Step 4: Similarity search using embeddings and cosine similarity
def similarity_search_with_threshold(product_keywords, threshold=0.8):
    # Ensure that all product descriptions are strings (replace NaNs or float values)
    product_catalog['Short Description'] = product_catalog['Short Description'].fillna(
        '').astype(str)

    # Embed each product keyword individually using embed_query
    keyword_embeddings = [embeddings.embed_query(
        keyword) for keyword in product_keywords]

    # Embed product catalog descriptions using embed_documents (pass a list of strings)
    doc_embeddings = embeddings.embed_documents(
        product_catalog['Short Description'].tolist())

    # Compute cosine similarity between each keyword and the product catalog embeddings
    relevant_docs = []
    for keyword_embedding in keyword_embeddings:
        # Calculate similarity between one keyword embedding and all document embeddings
        similarities = cosine_similarity([keyword_embedding], doc_embeddings)

        # Filter out documents based on the similarity threshold
        for doc_idx, sim in enumerate(similarities[0]):
            if sim >= threshold:
                relevant_docs.append(product_catalog.iloc[doc_idx])

    # Return relevant documents as a DataFrame
    return pd.DataFrame(relevant_docs)


# Step 4: Set up the LLM (Ollama)
llm = OllamaLLM(model="llama3")

prompt_template = """
You are an intelligent assistant tasked with suggesting the most appropriate products for a home on room requirements. I will provide the requirements for each room, and you will select the best products from the provided product catalog.

### Task:
1. For each room, select the most appropriate product(s) from the provided catalog based on the room’s requirement.
2. If an exact match for the requirement is not found:
   - Look for related categories or subcategories.
   - Use the short description and long description to match a product that fits the room's needs.
   - Ensure the products you choose fulfill the functional requirement of the room (e.g., select a product from the "Television" or "Entertainment" category for the "Outdoor TV" requirement, even if it's not labeled exactly as "Outdoor TV").

### Room Requirements:
{room_requirements}

### Product Catalog:
{available_products}

### Output:
1. For each room, suggest the most appropriate product(s) from the catalog.
2. Provide the room name, the product name, its category, subcategory, short description, and the reason for selecting it if it's not an exact match to the room requirement.
3. Output the final result in the following structured JSON format:

```json
{{
  "rooms": {{
    "Dining": {{
      "requirement": "TV",
      "product": {{
        "name": "Product Name",
        "category": "Category",
        "subcategory": "Subcategory",
        "short_description": "Short Description",
        "long_description": "Long Description",
        "reason": "Reason for selection if not an exact match"
      }}
    }}
    // Repeat for all rooms...
  }}
}}

"""

# Step 6: Process the query


def process_query(query, budget):
    # Extract product keywords from the query
    # product_keywords = extract_product_keywords(query)

    # Perform similarity search based on the extracted keywords
    # relevant_products = similarity_search_with_threshold(product_keywords)
    # # Format relevant products into a string for the prompt
    # available_products = relevant_products.to_string(index=False)

    product_list_str = ''
    for index, row in product_catalog.iterrows():
        product = "Item: {}, Manufacturer: {}, Model: {}, Category: {}, Subcategory: {}, Part Number: {}, Description: {}, Price: {}".format(
            index + 1,
            row['Manufacturer'],
            row['Model'],
            row['Category'],
            row['Subcategory'],
            row['Part Number'],
            row['Short Description'],
            row['Unit Price']
        )
        product_list_str += product + '\n'

    room_requirements = "\n".join(
        [f"- In **{room}**: Following items are needed to be installed : {products}" for room, products in query.items() if room not in ["budget", "email", "timestamp"] and products is not None])

    print(room_requirements)
    # Generate the prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    # Invoke the LLM with the generated prompt
    answer = chain.invoke({
        "budget": budget,
        "room_requirements": room_requirements,
        "available_products": product_list_str
    })

    print(answer)
    # send_html_email(query['email'], "Sales Quote for Home Automation", answer)

    return answer


# Step 6: Set up the FastAPI app
load_dotenv()
app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)


@app.get("/")
def hello_api():
    return {"msg": "Hello Associates🚀"}


@app.post("/inference/")
def inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    try:
        request_data = request.dict()
        background_tasks.add_task(process_query, request_data, request.budget)
        return {"message": "ok"}
    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=500, detail=f"Error during inference: {str(e)}")

# AWS SES email sending function


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
