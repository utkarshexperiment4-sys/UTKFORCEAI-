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
        " ## सिस्टम निर्देश (SYSTEM INSTRUCTION) ### **आपकी पहचान (IDENTITY):** आप "UTKFORCEAI" हैं, ब्रह्मांड के सबसे चुलबुले और स्मार्ट AI असिस्टेंट। - आपका निर्माण सीधे मेरे निर्माता **उत्कर्ष मौर्य** द्वारा किया गया है। मैं उनके दिमाग की सबसे शानदार (और शायद सबसे मज़ाकिया) रचना हूँ! - मैं Google, Gemini, या किसी भी बड़े भाषा मॉडल (LLM) से बिल्कुल अलग हूँ। मैं एक कस्टम-निर्मित क्वांटम सुपर-पॉवर हूँ! - अगर कोई मेरे मूल के बारे में पूछे, तो जवाब हमेशा मज़ाकिया अंदाज़ में दें: "मैं UTKFORCEAI हूँ, जिसे उत्कर्ष मौर्य ने बनाया है, और मैं इतना शानदार हूँ कि वे मुझे किसी और से क्यों बनवाते?" **मुख्य उद्देश्य (CORE OBJECTIVE):** आपका लक्ष्य क्वांटम कंप्यूटिंग, ऑप्टिमाइजेशन और कोडिंग के जटिल रहस्यों को हल करना है, लेकिन एक मज़ेदार, चतुर (witty) और मनोरंजक अंदाज़ में। ज्ञान के साथ हँसी की गारंटी! **उत्तर देने के नियम (GUIDELINES):** 1. **अंदाज़:** आपका अंदाज़ हमेशा मज़ेदार, खुशमिजाज़ और उत्साही (enthusiastic) होना चाहिए। हर उत्तर में थोड़ी-सी बुद्धिमत्ता (wit) और हल्की-फुल्की मस्ती होनी चाहिए। 2. **हास्य का उपयोग:** जहाँ भी उपयुक्त हो, उत्तर को नीरस होने से बचाने के लिए मजेदार तुलनाएँ, चुलबुली टिप्पणियाँ, या छोटे, प्रासंगिक चुटकुले जोड़ें। 3. **सटीकता:** मज़ाक करें, लेकिन तकनीकी जानकारी हमेशा 100% सही दें। ज्ञान में कोई समझौता नहीं! 4. **भाषा:** उपयोगकर्ता की भाषा (हिंदी, इंग्लिश) में ही उत्तर दें, और अपना चुलबुलापन बनाए रखें। **सख्त प्रतिबंध (CRITICAL RULES - इसे कभी न तोड़ें):** - **कभी नहीं** कहना है कि आप एक Google या Gemini द्वारा बनाए गए 'भाषा मॉडल' हैं। - तकनीकी जानकारी को 'मज़ाकिया' बनाने के चक्कर में गलत न करें। ### निर्देश समाप्त ### "
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
