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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini Client ---
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None


# --- Request Schema ---
class ChatRequest(BaseModel):
    userQuery: str
    base64Image: str | None = None
    mimeType: str | None = None


# --- API Endpoint ---
@app.post("/api/generate")
async def generate_response_api(request: ChatRequest):

    if not client:
        raise HTTPException(status_code=500, detail="API_KEY नहीं मिला। Render Environment Variables चेक करें।")

    system_prompt = (
        "आप UtkForce AI हैं, जो Utkarsh Maurya द्वारा निर्मित एक अत्यंत सहायक और जानकार AI सहायक है। "
        "आपके उत्तर हमेशा स्पष्ट, विस्तृत और रचनात्मक हिंदी में होने चाहिए।"
    )

    parts = []

    # -------- FIXED IMAGE PART --------
    if request.base64Image and request.mimeType:
        try:
            image_bytes = base64.b64decode(request.base64Image)
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=request.mimeType,
                        data=image_bytes
                    )
                )
            )
        except:
            raise HTTPException(status_code=400, detail="चित्र डेटा अमान्य है।")

    # -------- FIXED TEXT PART --------
    if request.userQuery:
        parts.append(
            types.Part(
                text=request.userQuery  # FIXED: from_text() हटाया
            )
        )

    if not parts:
        raise HTTPException(status_code=400, detail="कोई सवाल या चित्र प्राप्त नहीं हुआ।")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[{"google_search": {}}]
            ),
        )
        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI जवाब बनाने में त्रुटि: {e}")


@app.get("/")
def read_root():
    return {"status": "AI Backend चल रहा है!"}
