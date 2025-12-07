from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import requests
import difflib
# --- NEW IMPORTS FOR BIOBERT ---
from sentence_transformers import SentenceTransformer, util
import torch
import time

# ==========================
# CONFIGURATION
# ==========================
# ⚠️ PASTE YOUR ICD-11 KEYS HERE
CLIENT_ID = '0443b40a-5125-4241-b31b-b92c93122f54_a43ba522-ea21-497a-a7d0-27bc684143fe'
CLIENT_SECRET = 'DaGFcl18CyayKT9jE4Ol31H7P5RQy/Y7GrVcbqvm2QA='

app = FastAPI(title="SIH AYUSH-ICD Bridge", version="2.9")

# Enable CORS (Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# FAST DATA STORE (Indexed)
# ==========================
NAMASTE_DB = []
SEARCH_INDEX = {}  
SEARCH_KEYS = [] 

# --- BIOBERT MODEL & EMBEDDINGS ---
BIOBERT_MODEL = None
DB_EMBEDDINGS = None
DB_TEXTS = [] 

DATA_FILE = "namaste_data.json"
EMBEDDING_FILE = "embeddings.pt" # Cache file

def load_data():
    global NAMASTE_DB, SEARCH_INDEX, SEARCH_KEYS, BIOBERT_MODEL, DB_EMBEDDINGS, DB_TEXTS
    
    # 1. Load JSON Data
    if os.path.exists(DATA_FILE):
        try:
            print("⏳ Loading database...")
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                NAMASTE_DB = json.load(f)
            
            # Create Index for Instant Search
            for record in NAMASTE_DB:
                term = str(record.get('term', '')).lower().strip()
                english = str(record.get('english', '')).lower().strip()
                
                # Standard Search Index
                if term: 
                    SEARCH_INDEX[term] = record
                    SEARCH_KEYS.append(term)
                if english: 
                    SEARCH_INDEX[english] = record
                    SEARCH_KEYS.append(english)
                
                # Prepare text for semantic search
                definition = str(record.get('description', ''))
                semantic_text = f"{term} ({english}) - {definition}"
                DB_TEXTS.append({
                    "text": semantic_text, 
                    "record": record,
                    "clean_term": term 
                })
                
            print(f"✅ Database Loaded: {len(NAMASTE_DB)} records.")
            print(f"🚀 Search Index Built: {len(SEARCH_KEYS)} searchable keys.")
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            NAMASTE_DB = []
    else:
        print(f"⚠️ {DATA_FILE} not found. Server will start empty.")

    # 2. Load BioBERT Model
    try:
        print("🧠 Loading BioBERT Model...")
        BIOBERT_MODEL = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        # 3. Load or Generate Embeddings
        if os.path.exists(EMBEDDING_FILE):
            print("📂 Found cached embeddings. Loading from disk (FAST)...")
            DB_EMBEDDINGS = torch.load(EMBEDDING_FILE)
            print("✅ Embeddings Loaded!")
        else:
            print(f"⚙️ Generating Embeddings for {len(DB_TEXTS)} records (This takes time ONCE)...")
            start_time = time.time()
            
            # Extract texts
            corpus = [item["text"] for item in DB_TEXTS]
            
            # Encode (This is the slow part)
            DB_EMBEDDINGS = BIOBERT_MODEL.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
            
            # Save to cache
            torch.save(DB_EMBEDDINGS, EMBEDDING_FILE)
            print(f"💾 Embeddings saved to '{EMBEDDING_FILE}'. Next run will be instant!")
            print(f"⏱️ Time taken: {int(time.time() - start_time)} seconds")
            
        print("✅ Semantic Search Engine Ready!")
        
    except Exception as e:
        print(f"❌ Error loading BioBERT: {e}")
        BIOBERT_MODEL = None

# Load data on startup
load_data()

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_icd_token():
    url = "https://icdaccessmanagement.who.int/connect/token"
    payload = {
        'client_id': CLIENT_ID, 
        'client_secret': CLIENT_SECRET, 
        'grant_type': 'client_credentials', 
        'scope': 'icdapi_access'
    }
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code == 200:
            return resp.json().get('access_token')
    except Exception as e:
        print(f"Auth Error: {e}")
    return None

def search_who(term, token):
    """Step 1: Search for the term to get the URI"""
    url = "https://id.who.int/icd/release/11/2024-01/mms/search"
    headers = {
        'Authorization': f'Bearer {token}', 
        'Accept': 'application/json', 
        'Accept-Language': 'en', 
        'API-Version': 'v2'
    }
    params = {'q': term, 'useflexisearch': 'true'} 
    try:
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('destinationEntities'):
                return data['destinationEntities'][0]
    except Exception as e:
        print(f"WHO Search Error: {e}")
    return None

def fetch_icd_details(uri, token):
    """Step 2: Visit the specific URI to get the Definition"""
    headers = {
        'Authorization': f'Bearer {token}', 
        'Accept': 'application/json', 
        'Accept-Language': 'en', 
        'API-Version': 'v2'
    }
    try:
        resp = requests.get(uri, headers=headers)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Details Fetch Error: {e}")
    return None

def clean_html(text):
    if not text: return ""
    return text.replace("<em class='found'>", "").replace("</em>", "").replace("<p>", "").replace("</p>", "").strip()

# ==========================
# API ENDPOINTS
# ==========================

@app.get("/")
def home():
    ai_status = "Active" if BIOBERT_MODEL else "Inactive"
    return {
        "status": "Online", 
        "total_records": len(NAMASTE_DB), 
        "ai_engine": ai_status,
        "version": "2.9 (Cached Embeddings)"
    }

@app.get("/search")
def search_symptom(query: str):
    """
    Input: 'Kasa' or 'Cough' or 'breathing difficulty'
    Output: FHIR Compliant JSON with ICD-11 Mapping and Definitions
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = query.lower().strip()
    match = None
    search_method = "Unknown"

    # 1. INSTANT LOCAL SEARCH (Exact Match - Fastest)
    match = SEARCH_INDEX.get(query)
    if match: search_method = "Exact Match"

    # 2. SUBSTRING SEARCH (If exact match fails)
    if not match:
        match = next((item for item in NAMASTE_DB if query in str(item.get("term", "")).lower()), None)
        if match: search_method = "Substring Match"

    # 3. FUZZY SEARCH (Handles 
    #  mistakes)
    if not match:
        close_matches = difflib.get_close_matches(query, SEARCH_KEYS, n=1, cutoff=0.8) # stricter fuzzy
        if close_matches:
            match = SEARCH_INDEX.get(close_matches[0])
            search_method = f"Fuzzy Match ({close_matches[0]})"

    # 4. SEMANTIC AI SEARCH (BioBERT)
    if not match and BIOBERT_MODEL is not None and DB_EMBEDDINGS is not None:
        print(f"🧠 Running BioBERT Semantic Search for: '{query}'")
        
        # Encode user query
        query_embedding = BIOBERT_MODEL.encode(query, convert_to_tensor=True)
        
        # Find closest match
        hits = util.semantic_search(query_embedding, DB_EMBEDDINGS, top_k=1)
        
        if hits and hits[0]:
            best_hit = hits[0][0]
            score = best_hit['score']
            corpus_id = best_hit['corpus_id']
            
            # Threshold: Only accept if AI is reasonably confident
            if score > 0.45:
                match_data = DB_TEXTS[corpus_id]
                match = match_data['record']
                search_method = f"BioBERT AI (Confidence: {score:.2f})"
                print(f"   ✅ AI Found: '{match_data['text'][:50]}...'")

    if not match:
        raise HTTPException(status_code=404, detail=f"Symptom '{query}' not found in AYUSH Database (AI Search also failed).")

    print(f"🔍 Result found via: {search_method}")

    # ---------------------------------------------------------
    # BRIDGE LOGIC (Connect to WHO)
    # ---------------------------------------------------------
    token = get_icd_token()
    
    icd_code = "N/A"
    icd_title = "Connection Failed"
    icd_def = "" # Start empty
    icd_uri = ""

    if token:
        # Search WHO using the English term
        search_term = match.get('english')
        if not search_term or search_term.lower() == 'unknown':
            search_term = match.get('term')
            
        print(f"🌍 Searching WHO for: {search_term}")
        
        icd_result = search_who(search_term, token)
        
        if icd_result:
            icd_code = icd_result.get('theCode', 'Unspecified')
            icd_title = clean_html(icd_result.get('title', 'Unknown'))
            icd_uri = icd_result.get('id', '')
            
            # Fetch Definition
            if icd_uri:
                full_details = fetch_icd_details(icd_uri, token)
                if full_details:
                    raw_def = full_details.get('definition', None)
                    if isinstance(raw_def, dict):
                        icd_def = clean_html(raw_def.get('@value', ''))
                    elif isinstance(raw_def, str):
                        icd_def = clean_html(raw_def)
        else:
            icd_title = "No direct ICD-11 mapping found"

    # Fallback Description
    if not icd_def or icd_def == "No definition available.":
        local_desc = match.get('description', '')
        if local_desc and local_desc != "No description available.":
             icd_def = f"(Local Data): {local_desc}"
        else:
             icd_def = "No definition available."

    # 3. Construct FHIR-Compliant Response
    return {
        "resourceType": "Condition",
        "clinicalStatus": "active",
        "verificationStatus": "provisional",
        "meta": {
            "searchMethod": search_method
        },
        "code": {
            "coding": [
                {
                    "system": match.get('system', 'Ayurveda'),
                    "code": match.get('tm2_code', 'Unknown'),
                    "display": match.get('term', 'Unknown'),
                    "definition": match.get('description', 'No description available.')
                },
                {
                    "system": "http://id.who.int/icd/release/11/mms",
                    "code": icd_code,
                    "display": icd_title,
                    "uri": icd_uri,
                    "definition": icd_def
                }
            ],
            "text": f"{match.get('term')} (English: {match.get('english')}) mapped to {icd_title}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow access from other devices if needed
    uvicorn.run(app, host="127.0.0.1", port=8000)