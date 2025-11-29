from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import google.generativeai as genai
import uvicorn
from typing import Optional
import os
from fastapi.middleware.cors import CORSMiddleware

# ============================
# CONFIGURATION - Using Environment Variables
# ============================
CODA_API_KEY = os.getenv("CODA_API_KEY", "47b5abed-b00e-4029-a88b-da2e576eae36")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDxkiMGS2AkjVsaL4DrI5Q7BeEoyh93kt0")
BASE_URL = "https://coda.io/apis/v1"
CODA_HEADERS = {"Authorization": f"Bearer {CODA_API_KEY}"}

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="Coda AI Assistant",
    description="AI-powered assistant for querying Coda data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# DATA MODELS
# ============================
class CliqRequest(BaseModel):
    text: str
    user_name: Optional[str] = "User"
    user_id: Optional[str] = None
    channel_name: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str

class AIResponse(BaseModel):
    answer: str
    status: str = "success"

# ============================
# CODA API FUNCTIONS
# ============================
def get_all_docs():
    url = f"{BASE_URL}/docs"
    res = requests.get(url, headers=CODA_HEADERS)
    return res.json().get("items", [])

def resolve_doc_id(doc_name):
    for d in get_all_docs():
        if d["name"].strip().lower() == doc_name.strip().lower():
            return d["id"]
    return None

def get_pages(doc_id):
    url = f"{BASE_URL}/docs/{doc_id}/pages"
    res = requests.get(url, headers=CODA_HEADERS)
    return res.json().get("items", [])

def get_tables_created_on_page(doc_id, page_id):
    url = f"{BASE_URL}/docs/{doc_id}/tables"
    res = requests.get(url, headers=CODA_HEADERS)
    all_tables = res.json().get("items", [])
    
    page_specific_tables = []
    for table in all_tables:
        if table.get('parent', {}).get('id') == page_id:
            page_specific_tables.append(table)
    
    return page_specific_tables

def get_column_map(doc_id, table_id):
    url = f"{BASE_URL}/docs/{doc_id}/tables/{table_id}/columns"
    res = requests.get(url, headers=CODA_HEADERS)
    cols = res.json().get("items", [])
    return {c["id"]: c["name"] for c in cols}

def get_rows(doc_id, table_id, colmap):
    url = f"{BASE_URL}/docs/{doc_id}/tables/{table_id}/rows"
    res = requests.get(url, headers=CODA_HEADERS)
    items = res.json().get("items", [])

    readable_rows = []
    for row in items:
        clean = {}
        for col_id, val in row["values"].items():
            clean[colmap.get(col_id, col_id)] = val
        readable_rows.append(clean)

    return readable_rows

# Cache for Coda data
coda_data_cache = None
formatted_data_cache = None

def get_all_coda_data(doc_id):
    """Fetch all data from Coda and format it for Gemini"""
    global coda_data_cache
    
    if coda_data_cache is not None:
        return coda_data_cache
        
    pages = get_pages(doc_id)
    all_data = []
    
    for page in pages:
        page_name = page["name"]
        page_id = page["id"]
        
        page_data = {
            "page_name": page_name,
            "tables": []
        }
        
        tables = get_tables_created_on_page(doc_id, page_id)
        
        for table in tables:
            table_name = table["name"]
            table_id = table["id"]
            
            try:
                colmap = get_column_map(doc_id, table_id)
                rows = get_rows(doc_id, table_id, colmap)
                
                # Filter out empty rows
                filtered_rows = []
                for r in rows:
                    has_data = any(v for v in r.values() if v not in [None, "", False, "False"])
                    if has_data:
                        filtered_rows.append(r)
                
                if filtered_rows:
                    table_data = {
                        "table_name": table_name,
                        "columns": list(filtered_rows[0].keys()) if filtered_rows else [],
                        "rows": filtered_rows
                    }
                    page_data["tables"].append(table_data)
                    
            except Exception as e:
                print(f"Error processing table {table_name}: {str(e)}")
                continue
        
        if page_data["tables"]:
            all_data.append(page_data)
    
    coda_data_cache = all_data
    return all_data

def format_data_for_gemini(coda_data):
    """Format the Coda data into a readable string for Gemini"""
    global formatted_data_cache
    
    if formatted_data_cache is not None:
        return formatted_data_cache
        
    formatted_text = "CODA DOCUMENT DATA:\n\n"
    
    for page in coda_data:
        formatted_text += f"=== PAGE: {page['page_name']} ===\n"
        
        for table in page['tables']:
            formatted_text += f"\nTABLE: {table['table_name']}\n"
            formatted_text += f"Columns: {', '.join(table['columns'])}\n"
            
            for i, row in enumerate(table['rows'], 1):
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v not in [None, "", False, "False"]])
                formatted_text += f"{i}. {row_text}\n"
            
            formatted_text += "\n"
    
    formatted_data_cache = formatted_text
    return formatted_text

# ============================
# GEMINI AI FUNCTIONS
# ============================
def ask_gemini_about_data(context_data, question):
    """Ask Gemini AI a question about the Coda data"""
    
    prompt = f"""
    You are an AI assistant analyzing data from a Coda document. 
    Here is the data from the Coda document:

    {context_data}

    Please analyze this data and answer the following question:
    {question}

    Instructions:
    1. Be specific and accurate in your answer
    2. Reference actual data points from the tables when possible
    3. If the data doesn't contain information to answer the question, say so
    4. Provide insights based on the available data
    5. Keep your response clear and concise
    6. Format the response in a readable way without using markdown symbols like *

    Answer:
    """
    
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error communicating with Gemini AI: {str(e)}"

# ============================
# FASTAPI ROUTES
# ============================
@app.get("/")
async def root():
    return {
        "message": "Coda AI Assistant API is running!", 
        "status": "active",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Coda AI Assistant"}

@app.post("/ask", response_model=AIResponse)
async def ask_question(request: QuestionRequest):
    """API endpoint to ask questions about Coda data"""
    try:
        doc_name = "samdanielvincy's Coda Playground"
        doc_id = resolve_doc_id(doc_name)
        
        if not doc_id:
            raise HTTPException(status_code=404, detail="Coda document not found")
        
        # Get Coda data
        coda_data = get_all_coda_data(doc_id)
        formatted_data = format_data_for_gemini(coda_data)
        
        # Get AI response
        answer = ask_gemini_about_data(formatted_data, request.question)
        
        return AIResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/cliq/ask")
async def cliq_webhook(request: CliqRequest):
    """Webhook endpoint for Zoho Cliq slash commands"""
    try:
        if not request.text.strip():
            return {
                "text": "Please provide a question. Usage: /coda-ai <your question>",
                "color": "#FF0000"
            }
        
        doc_name = "samdanielvincy's Coda Playground"
        doc_id = resolve_doc_id(doc_name)
        
        if not doc_id:
            return {
                "text": "❌ Coda document not found. Please check the configuration.",
                "color": "#FF0000"
            }
        
        # Get Coda data
        coda_data = get_all_coda_data(doc_id)
        formatted_data = format_data_for_gemini(coda_data)
        
        # Get AI response
        answer = ask_gemini_about_data(formatted_data, request.text)
        
        # Format response for Cliq
        response_text = f"🤖 **Coda AI Assistant**\n\n**Question:** {request.text}\n\n**Answer:** {answer}"
        
        return {
            "text": response_text,
            "color": "#4CAF50"
        }
        
    except Exception as e:
        return {
            "text": f"❌ Error processing your question: {str(e)}",
            "color": "#FF0000"
        }

@app.post("/refresh-cache")
async def refresh_cache():
    """Force refresh the Coda data cache"""
    global coda_data_cache, formatted_data_cache
    coda_data_cache = None
    formatted_data_cache = None
    
    doc_name = "samdanielvincy's Coda Playground"
    doc_id = resolve_doc_id(doc_name)
    
    if doc_id:
        get_all_coda_data(doc_id)  # This will refresh the cache
        return {"message": "Cache refreshed successfully", "status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Coda document not found")

@app.get("/data-summary")
async def get_data_summary():
    """Get summary of available Coda data"""
    try:
        doc_name = "samdanielvincy's Coda Playground"
        doc_id = resolve_doc_id(doc_name)
        
        if not doc_id:
            raise HTTPException(status_code=404, detail="Coda document not found")
        
        coda_data = get_all_coda_data(doc_id)
        
        total_pages = len(coda_data)
        total_tables = sum(len(page['tables']) for page in coda_data)
        total_rows = sum(len(table['rows']) for page in coda_data for table in page['tables'])
        
        pages_info = []
        for page in coda_data:
            page_info = {
                "page_name": page['page_name'],
                "tables": [
                    {
                        "table_name": table['table_name'],
                        "row_count": len(table['rows'])
                    }
                    for table in page['tables']
                ]
            }
            pages_info.append(page_info)
        
        return {
            "document": doc_name,
            "summary": {
                "total_pages": total_pages,
                "total_tables": total_tables,
                "total_rows": total_rows
            },
            "pages": pages_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data summary: {str(e)}")

# ============================
# RUN THE APPLICATION
# ============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
