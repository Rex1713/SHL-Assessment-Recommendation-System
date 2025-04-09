from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.llms.groq import Groq
from llama_index.core.settings import Settings
from llama_index.core.response_synthesizers import CompactAndRefine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM + embedding setup
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
Settings.llm = Groq(model=model_name, api_key=GROQ_API_KEY)
Settings.embed_model = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-large-en-v1.5")

# FastAPI app
app = FastAPI(title="SHL Assessment Recommender API")

# Pydantic model for input
class QueryRequest(BaseModel):
    query: str  # can be raw text or a URL

# Extract text from URL if necessary
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        return text.strip()
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

# Load SHL data
def load_shl_data_with_metadata(csv_path: str):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        assessment_name = str(row['Assessment Name'])
        description = str(row['Description'])
        assessment_type = str(row['Types']).strip()
        raw_duration = str(row['Assessment Length (minutes)']).strip()
        remote = str(row['Remote Support']).strip()
        adaptive = str(row['Adaptive Support']).strip()
        job_levels = str(row['Job Levels']).strip()
        url = str(row['URL']).strip()

        try:
            minutes = int(float(raw_duration))
            if minutes == 9999:
                duration_clean = "Untimed"
            elif minutes == -1:
                duration_clean = "Variable"
            else:
                duration_clean = f"{minutes} minutes"
        except ValueError:
            duration_clean = "Variable"
            minutes = -1

        text = f"""
        Assessment: {assessment_name}
        Description: {description}
        Type: {assessment_type}
        Duration: {duration_clean}
        Remote: {remote}
        Adaptive: {adaptive}
        Job Levels: {job_levels}
        URL: {url}
        """

        metadata = {
            "assessment_name": assessment_name,
            "type": assessment_type,
            "duration_minutes": minutes,
            "remote": remote,
            "adaptive": adaptive,
            "job_levels": job_levels,
            "url": url,
            "description": description
        }

        node = TextNode(text=text.strip(), metadata=metadata)
        documents.append(node)

    return documents

# Load or initialize index
csv_path = "data/shl_product_catalog.csv"
persist_dir = "shl_index"
if not os.path.exists(persist_dir):
    nodes = load_shl_data_with_metadata(csv_path)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="compact_and_refine",
    response_synthesizer=CompactAndRefine()
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Recommend endpoint with POST method
@app.post("/recommend")
async def recommend_assessments(req: QueryRequest):
    query = req.query

    if query.startswith("http://") or query.startswith("https://"):
        query = extract_text_from_url(query)
        if query.startswith("Error"):
            return {"error": query}

    # Perform the query
    response = query_engine.query(query)
    records = []

    for node in response.source_nodes:
        meta = node.node.metadata
        duration = meta.get("duration_minutes", 10)
        try:
            duration = int(duration)
        except:
            duration = 10

        records.append({
            "url": meta.get("url", ""),
            "adaptive_support": "Yes" if str(meta.get("adaptive", "")).strip().lower() == "yes" else "No",
            "description": meta.get("description", ""),
            "duration": max(1, duration),
            "remote_support": "Yes" if str(meta.get("remote", "")).strip().lower() == "yes" else "No",
            "test_type": [t.strip() for t in str(meta.get("type", "")).split(",") if t.strip()]
        })

    return {
        "recommended_assessments": records[:10]
    }
