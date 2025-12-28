import requests
from PIL import Image, ImageDraw, ImageFont
import io
import json
import pandas as pd
import os

def create_test_image():
    # Create an image 300x100
    img = Image.new('RGB', (300, 100), color='white')
    d = ImageDraw.Draw(img)
    # Don't worry about font for now, PIL default might be too small/unreadable for defaults?
    # Actually, default PIL font works for English. For Korean we need a font...
    # Let's just test English first to verify the pipeline.
    # Left side (0-150): "Apple"
    d.text((10, 40), "Apple", fill=(0,0,0))
    # Right side (150-300): "BigFruit" (Simulated meaning)
    d.text((160, 40), "BigFruit", fill=(0,0,0))
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def test_upload():
    url = "http://localhost:8000/upload"
    img_buf = create_test_image()
    
    # Split at x=150
    splits = json.dumps([150])
    
    files = {'file': ('test.png', img_buf, 'image/png')}
    data = {'splits': splits}
    
    print("Sending request...")
    res = requests.post(url, files=files, data=data)
    print("Status:", res.status_code)
    print("Response:", res.json())
    
    if res.status_code == 200:
        download_url = res.json()['download_url']
        excel_url = f"http://localhost:8000{download_url}"
        
        # Download Excel
        print("Downloading Excel...")
        xl_res = requests.get(excel_url)
        with open("test_result.xlsx", "wb") as f:
            f.write(xl_res.content)
            
        print("Reading Excel...")
        df = pd.read_excel("test_result.xlsx")
        print("Data:\n", df)
        
        # Verify columns
        if "Word" in df.columns and "Meaning" in df.columns:
            print("SUCCESS: Columns found.")
            # Simple check content roughly (OCR might fail on default font/size, so we're testing pipeline largely)
        else:
            print("FAILURE: Columns missing.")

if __name__ == "__main__":
    try:
        test_upload()
    except Exception as e:
        print("Test failed:", e)
