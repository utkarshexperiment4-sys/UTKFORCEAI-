# --- जरूरी Imports ---
import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- FastAPI App और CORS सेटअप ---
app = FastAPI()

# CORS Setup: 'https://utkarshexperiment4-sys.github.io' को आपके डोमेन से बदलें
origins = [
    "https://utkarshexperiment4-sys.github.io", 
    "https://utkarshexperiment4-sys.github.io/UTKFORCEAI/",  # <-- यह नया है
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Gemini Client सेटअप ---
API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=API_KEY) if API_KEY else None
except Exception:
    client = None

# --- रिक्वेस्ट स्कीमा ---
class ChatRequest(BaseModel):
    userQuery: str
    base64Image: str | None = None
    mimeType: str | None = None

# --- API Endpoint (यही वह जगह है जहाँ आपने दिया गया कोड जाएगा) ---
@app.post("/api/generate")
async def generate_response_api(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="AI सेवा शुरू नहीं हुई।")

    # **********************************************
    # *** आपने दिया गया कोड यहाँ से शुरू होता है ***
    # **********************************************

    system_prompt = (
        f"आप UtkForce AI हैं, जो Utkarsh Maurya द्वारा निर्मित एक अत्यंत सहायक, जानकार और विनम्र AI सहायक है। "
        f"आप दुनिया की जानकारी तक पहुँच सकते हैं, चित्र समझ सकते हैं, और हमेशा विस्तृत, तथ्यात्मक रूप से सही और रचनात्मक उत्तर हिंदी में प्रदान करेंगे। "
        f"आपके उत्तरों में 'हाँ' या 'ना' शब्द या उनके पर्यायवाची का उपयोग बिल्कुल नहीं होना चाहिए। "
        f"उत्तर को साफ-सुथरे पैराग्राफ या बुलेट पॉइंट्स में फ़ॉर्मेट करें।"
    )
    
    parts = []
    
    if request.base64Image and request.mimeType:
        try:
            image_bytes = base64.b64decode(request.base64Image)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=request.mimeType))
        except Exception:
            raise HTTPException(status_code=400, detail="प्रदान किया गया चित्र डेटा अमान्य है।")

    if request.userQuery:
        parts.append(types.Part.from_text(request.userQuery))
    
    if not parts:
        raise HTTPException(status_code=400, detail="कोई सवाल या चित्र प्रदान नहीं किया गया।")

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[{"google_search": {}}]
            )
        )
        
        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI जवाब जेनरेट करने में त्रुटि: {e}")

    # **********************************************
    # *** आपने दिया गया कोड यहाँ समाप्त होता है ***
    # **********************************************

# --- हेल्थ चेक रूट (ज़रूरी नहीं, पर अच्छा है) ---
@app.get("/")
def read_root():
    return {"status": "AI Backend चल रहा है।"}

