import os
import shutil
import uuid
import json
import io
import requests
from typing import List, Optional, Tuple, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pytesseract
from PIL import Image
import pandas as pd
import uvicorn
import imagehash
import numpy as np

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/output", StaticFiles(directory="output"), name="output")

# Ensure directories exist
os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)

SPLITS_DB_FILE = "splits_db.json"
CORRECTIONS_DB_FILE = "corrections_db.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# --- DB Helper Functions ---

def load_json_db(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_json_db(filepath, db):
    with open(filepath, "w") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_splits_db(): return load_json_db(SPLITS_DB_FILE)
def save_splits_db(db): save_json_db(SPLITS_DB_FILE, db)

def load_corrections_db(): return load_json_db(CORRECTIONS_DB_FILE)
def save_corrections_db(db): save_json_db(CORRECTIONS_DB_FILE, db)

# --- Logic Functions ---

def get_image_hash(img: Image.Image) -> str:
    return str(imagehash.dhash(img))

def find_best_splits(current_hash_str: str, db: dict, threshold: int = 10) -> Optional[List[float]]:
    if not db: return None
    current_hash = imagehash.hex_to_hash(current_hash_str)
    best_dist = float('inf')
    best_splits = None
    
    for saved_hash_str, splits in db.items():
        saved_hash = imagehash.hex_to_hash(saved_hash_str)
        dist = current_hash - saved_hash
        if dist < best_dist:
            best_dist = dist
            best_splits = splits
            
    if best_dist <= threshold:
        return best_splits
    return None

def apply_corrections(text: str, db: dict) -> str:
    # We iterate line by line to preserve structure.
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
            
        # Split into words (whitespace)
        words = line.split()
        new_words = []
        for w in words:
            val = db.get(w, w)
            new_words.append(val)
        
        new_lines.append(" ".join(new_words))
        
    return "\n".join(new_lines)

def align_with_llm(headers: List[str], columns_data: List[List[str]]) -> List[Dict[str, str]]:
    """
    Uses Ollama to semantic align columns.
    """
    # Construct prompt
    # columns_data is [[word1, word2...], [meaning1, meaning2...]]
    
    # We need to present this clearly to the LLM.
    # Let's clean up empty strings first?
    # Actually, empty strings might be gaps. But OCR usually just gives raw lines.
    
    prompt_data = {}
    for i, h in enumerate(headers):
        # Filtering out empty lines might help LLM focus, OR it might lose structural hints.
        # Let's filter empty.
        prompt_data[h] = [x for x in columns_data[i] if x.strip()]

    prompt = f"""
    You are a data processing assistant. I have performed OCR on a vocabulary list but the columns are misaligned.
    I typically have a 'Word' column and a 'Meaning' or 'Example' column.
    
    Here is the raw data extracted from vertical columns:
    {json.dumps(prompt_data, ensure_ascii=False, indent=2)}
    
    Current Problem: The number of lines may not match, or multi-line meanings may have been split.
    Task: Align these lists into a single logical list of Vocabulary Items.
    
    IMPORTANT RULES:
    1. Match meaning/example text to the correct Word.
    2. The 'Meaning' column often starts with Part-of-Speech tags like 'n.', 'v.', 'phr.', 'adv.', 'adj.'. Use these tags as STRONG INDICATORS that a new definition is starting.
    3. If a meaning spans multiple lines (and doesn't start with a new POS tag), it likely belongs to the previous word. Combine it.
    4. Return a JSON Array of objects. Each object should have keys: {', '.join(headers)}.
    5. Output ONLY valid JSON. No markdown, no explanations.
    """
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json" 
        }
        
        print(f"DEBUG: Sending request to Ollama ({OLLAMA_API_URL})...")
        print(f"DEBUG: Payload model={OLLAMA_MODEL}")
        
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=120) # Increased timeout
        print(f"DEBUG: Ollama response status: {res.status_code}")
        
        res.raise_for_status()
        
        result = res.json()
        response_text = result.get("response", "")
        print(f"DEBUG: Ollama raw response: {response_text[:200]}...") # Print first 200 chars
        
        # Parse JSON
        aligned_data = json.loads(response_text)
        print("DEBUG: Successfully parsed JSON from Ollama.")
        
        # Ensure it's a list
        if isinstance(aligned_data, dict):
            # Sometimes model wraps it in {"data": [...]}
            for k, v in aligned_data.items():
                if isinstance(v, list):
                    return v
            return [aligned_data] # Fallback
            
        return aligned_data
        
    except requests.exceptions.Timeout:
        print("ERROR: Ollama request timed out.")
        # Fallback
    except Exception as e:
        print(f"ERROR: LLM Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        max_len = max(len(col) for col in columns_data)
        fallback = []
        for i in range(max_len):
            row = {}
            for idx, h in enumerate(headers):
                if i < len(columns_data[idx]):
                    row[h] = columns_data[idx][i]
                else:
                    row[h] = ""
            fallback.append(row)
        return fallback

# --- Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/suggest-splits")
async def suggest_splits(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        h = get_image_hash(img)
        db = load_splits_db()
        suggested = find_best_splits(h, db)
        return JSONResponse({"status": "success", "splits": suggested or []})
    except Exception as e:
        print(f"Suggestion error: {e}")
        return JSONResponse({"status": "error", "splits": []})

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    splits: str = Form("[]") 
):
    try:
        split_coords = json.loads(splits) 
        split_coords = [int(x) for x in split_coords]
        split_coords.sort()

        # Save Image
        file_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1]
        input_filename = f"{file_id}.{file_ext}"
        input_path = os.path.join("images", input_filename)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(input_path)
        width, height = img.size
        
        # Learning Splits
        h = get_image_hash(img)
        split_db = load_splits_db()
        if width > 0:
            splits_pct = [x / width for x in split_coords]
        else:
            splits_pct = []
        
        if splits_pct:
            split_db[h] = splits_pct
            save_splits_db(split_db)
            
        # Processing
        corrections_db = load_corrections_db()
        columns_data = [] 
        headers = [] # Initialize headers
        
        if not split_coords:
            # Fallback for no splits (treat as single block)
            text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 6')
            corrected_text = apply_corrections(text, corrections_db)
            lines = [line.strip() for line in corrected_text.splitlines() if line.strip()]
            columns_data.append(lines)
            headers = ["Extracted Text"]
        else:
            boundaries = [0] + split_coords + [width]
            
            for i in range(len(boundaries) - 1):
                start_x = boundaries[i]
                end_x = boundaries[i+1]
                if end_x - start_x < 5: continue 
                
                crop_box = (start_x, 0, end_x, height)
                cropped_img = img.crop(crop_box)
                
                # Run OCR
                text = pytesseract.image_to_string(cropped_img, lang='kor+eng', config='--psm 6')
                
                # Correction
                text = apply_corrections(text, corrections_db)
                
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                columns_data.append(lines)
                
                if i == 0: headers.append("Word")
                elif i == 1: headers.append("Meaning")
                else: headers.append(f"Column {i+1}")
        
        # LLM ALIGNMENT
        # We only try alignment if we have multiple columns (more than 1)
        if len(headers) > 1:
            aligned_rows = align_with_llm(headers, columns_data)
        else:
            # Single column, just wrap in dict
            aligned_rows = [{headers[0]: line} for line in columns_data[0]]

        return JSONResponse({
            "status": "success",
            "file_id": file_id,
            "headers": headers,
            "data": aligned_rows,
            "original_for_learning": aligned_rows 
        })

    except Exception as e:
        print(f"Error processing: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/save-result")
async def save_result(
    file_id: str = Body(...),
    corrected_data: List[Dict[str, str]] = Body(...),
    original_data: List[Dict[str, str]] = Body(...)
):
    try:
        # 1. Learning Logic 
        corrections_db = load_corrections_db()
        count_learned = 0
        
        min_len = min(len(corrected_data), len(original_data))
        
        for i in range(min_len):
            orig_row = original_data[i]
            corr_row = corrected_data[i]
            
            for key in orig_row:
                orig_val = orig_row.get(key, "").strip()
                corr_val = corr_row.get(key, "").strip()
                
                if orig_val and corr_val and orig_val != corr_val:
                    corrections_db[orig_val] = corr_val
                    count_learned += 1
                    
        if count_learned > 0:
            save_corrections_db(corrections_db)
            print(f"Learned {count_learned} new corrections.")

        # 2. Generate Excel
        df = pd.DataFrame(corrected_data)
        excel_filename = f"vocab_{file_id}.xlsx"
        excel_path = os.path.join("output", excel_filename)
        df.to_excel(excel_path, index=False)
        
        return JSONResponse({
            "status": "success",
            "download_url": f"/download/{excel_filename}"
        })

    except Exception as e:
        print(f"Save Result Error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
