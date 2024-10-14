from langchain_huggingface import HuggingFaceEmbeddings
import openai
import tiktoken
import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Load environment variables (e.g., OpenAI API key, AWS credentials)
load_dotenv()

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the product catalog
file_path = './product_catalog.csv'  # Change this to the actual path
product_catalog = pd.read_csv(file_path)

# Extract only the columns we care about
product_catalog = product_catalog[['Manufacturer', 'Model', 'Category',
                                   'Subcategory', 'Part Number', 'Short Description', 'Unit Price']]

# Initialize HuggingFace Embeddings (optional)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Tokenizer model setup for token counting
# Adjust this to the model you're using
encoding = tiktoken.encoding_for_model("gpt-4")

# Step 3: Extract keywords from the room requirements


def extract_product_keywords(query):
    keywords = []
    for room, products in query.items():
        if room not in ["budget", "email", "timestamp"]:
            keywords.extend([product.strip()
                            for product in products.split(",")])
    return keywords

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

# Function to count tokens


def count_tokens(prompt, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt)
    return len(tokens)


# Prompt template for generating the sales quote
prompt_template = """
You are a sales assistant for home automation products. Based on the customer's requirements and budget, generate a detailed sales quotation that includes three tiers: Normal, Good, and Best.

**Customer Requirements**:
- Budget: {budget}
- Requirements per room:
{room_requirements}

For each room, search the product catalog and find relevant products that match the customer’s requirements. The available product list is provided below. Ensure that the products you select fit within the specified budget for each of the three tiers.

### Available Products:
{available_products}

### Output:
Please generate the sales quotation in an HTML-compatible tabular format that includes:
- Room
- Manufacturer
- Model or Part Number
- Product Name
- Quantity
- Unit Price (in Pounds)
- Total Price (in Pounds)

Ensure that the output is returned as valid HTML code.
"""

# Function to generate the sales quote using ChatGPT


# Function to generate the sales quote using ChatGPT (chat-based API)
def generate_quote_with_chatgpt(budget, room_requirements, available_products):
    # Format the prompt
    prompt = prompt_template.format(
        budget=budget,
        room_requirements=room_requirements,
        available_products=available_products
    )

    # Count tokens in the prompt
    prompt_tokens = count_tokens(prompt, model="gpt-4")
    print(f"Prompt tokens: {prompt_tokens}")

    # Calculate how many tokens are left for the response
    # Adjust this based on the model you're using (e.g., 4096 for gpt-3.5-turbo)
    max_model_tokens = 8192
    max_tokens_for_response = max_model_tokens - prompt_tokens

    # Ensure we don't exceed the model's token limit
    if max_tokens_for_response <= 0:
        raise ValueError("Prompt is too long for the model's token limit.")

    # Prepare the messages for the chat API
    messages = [
        {"role": "system", "content": "You are a helpful sales assistant for home automation."},
        {"role": "user", "content": prompt}
    ]

    # Call OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" based on your access
            messages=messages,
            max_tokens=max_tokens_for_response,
            temperature=0.7
        )

        # Extract the response text
        return response.choices[0].message['content'].strip()

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None
# Step 6: Process the query


def process_query(query, budget):
    # Extract product keywords from the query
    product_keywords = extract_product_keywords(query)

    # Perform similarity search based on the extracted keywords
    relevant_products = similarity_search_with_threshold(product_keywords)
    # Format relevant products into a string for the prompt
    available_products = relevant_products.to_string(index=False)

    room_requirements = "\n".join(
        [f"- {room}: {products}" for room, products in query.items() if room not in ["budget", "email", "timestamp"]])

    # Use the OpenAI API to generate the sales quotation
    answer = generate_quote_with_chatgpt(
        budget, room_requirements, available_products)

    # Send the email with the generated quote
    if answer:
        send_html_email(
            query['email'], "Sales Quote for Home Automation", answer)

    return answer


# Step 6: Set up the FastAPI app
app = FastAPI(title="Sales Quotation API", version="1.0")

# Endpoint to test if the API is working


@app.get("/")
def hello_api():
    return {"msg": "Hello Associates🚀"}

# Inference request model


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

# API endpoint to trigger the inference


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
