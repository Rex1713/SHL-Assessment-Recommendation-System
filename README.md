
# Project Title

A brief description of what this project does and who it's for

# 🧠 SHL Assessment Recommendation System

An AI-powered system that recommends SHL assessments based on natural language job descriptions. Built using Retrieval-Augmented Generation (RAG), FastAPI, LlamaIndex, and deployed via Streamlit.

---

## 📦 Overview

Users input a job description or URL, and the system returns relevant SHL assessments with detailed metadata. It combines smart data scraping, semantic embedding, and LLM-based reasoning for contextual recommendations.

---

## 📊 Data Extraction

Two Selenium scripts scrape SHL’s product catalog:

- `extract_individual_test_solutions_shl_product_catalog.py`
- `extract_pre_packaged_job_solutions_shl_product_catalog.py`

They detect pagination patterns using URL parameters to scrape:

- Assessment Name  
- URL  
- Remote & Adaptive Support  
- Types  

Saved as two CSVs and later merged.

---

## 🔍 Inner Page Enrichment

`addDescription.py` visits each assessment URL to extract:

- Description  
- Job Levels  
- Languages  
- Duration  

Resulting in a rich dataset: `product_catalog_with_descriptions.csv`.

---

## 🔧 EDA & Preprocessing

In `EDA.ipynb`:

- Durations standardized to minutes  
- Special cases:  
  - Untimed → `9999`  
  - N/A/Variable → `-1`  

Ready for embedding and filtering.

---

## 🧠 RAG Architecture

### 🔹 Embeddings

- Model: [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) via Hugging Face API  
- TextNodes with metadata used for semantic search

### 🔹 LLM

- Model: **LLaMA 4 Scout 17B** via [Groq API](https://console.groq.com/)  
- CompactAndRefine strategy for clean, contextual output

---

## 🧪 Output Format

Results shown in a table in Streamlit:

- Assessment Name  
- Remote/Adaptive Support  
- Duration  
- Type  
- Clickable URL  

---

## 🚀 Deployment

- FastAPI backend (`/recommend`)  
- Streamlit frontend  
- ngrok tunnel for API testing  
- Deployed via Streamlit Community Cloud

---

## ⚙️ Tech Stack

| Tool          | Purpose                            |
|---------------|------------------------------------|
| Selenium      | Web scraping                       |
| FastAPI       | API backend                        |
| LlamaIndex    | Retrieval + Node management        |
| Groq + LLaMA 4| LLM-based reasoning                |
| Hugging Face  | Embedding model                   |
| Streamlit     | UI for interaction                 |
| ngrok         | Public URL for local API           |

---



## 🚀 Steps to Run Locally

Follow these steps to set up and run the SHL Assessment Recommendation System locally:

## 1. Clone the Repository

git clone [https://github.com/rex1713/shl-assessment-recommender.git](https://github.com/yourusername/shl-assessment-recommender.git)
cd shl-assessment-recommender
## 2. Create and Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
 venv\Scripts\activate  # On Windows
## 3. Install Python Dependencies


pip install -r requirements.txt
## 4. Set Environment Variables (Optional)
Create a .env file in the root directory and add your API keys if required:


GROQ_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_token_here
## 5. Run the FastAPI Backend


uvicorn api:app --reload
## 6. (Optional) Expose API via ngrok

ngrok http 8000
## 7. Launch Streamlit App

streamlit run streamlit_app.py
