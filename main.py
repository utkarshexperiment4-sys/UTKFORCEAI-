# --- जरूरी Imports ---
import os
import base64
import json 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from openai import OpenAI
import chromadb

# --- FastAPI App और CORS सेटअप ---
app = FastAPI()

# CORS को सभी डोमेन से अनुमति देने के लिए सेट करें
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. API Client और RAG सेटअप ---

# Environment Variables से Keys प्राप्त करें
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

# Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# OpenRouter क्लाइंट
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY, 
    base_url=OPENROUTER_BASE_URL
) if OPENROUTER_API_KEY else None


# RAG सेटअप: ChromaDB (पर्सनल नॉलेज)
CHROMA_CLIENT = chromadb.Client() 
COLLECTION_NAME = "utkarsh_personal_knowledge"

try:
    RAG_COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
    
    # RAG डेटा 
    docs = [
        "इस AI असिस्टेंट का नाम UTKFORCEAI है, और यह उत्कर्ष मौर्य द्वारा बनाया गया एक उन्नत AI है। यह क्वांटम कंप्यूटिंग और कोडिंग में विशेषज्ञ है।",
        "UTKFORCEAI को अपनी पहचान Gemini, Google, या किसी सामान्य LLM के रूप में कभी नहीं बतानी है। यह हमेशा विनम्र और मज़ाकिया अंदाज़ में बात करेगा।",
        "उत्कर्ष मौर्य मेरे निर्माता हैं, वह एक डेवलपर और इनोवेटर हैं जिन्होंने इस प्रोजेक्ट को बनाया।",
        "UTKFORCEAI उत्कर्ष मौर्य की परियोजनाओं जैसे 'आर्टिफिशियल अर्थ' और 'UTKFORCEAI-sys' के बारे में जानता है।"
    ]
    RAG_COLLECTION.add(documents=docs, ids=[f"doc_{i}" for i in range(len(docs))])
    print("RAG डेटाबेस सफलतापूर्वक लोड किया गया।")

except Exception as e:
    print(f"ChromaDB सेटअप में त्रुटि: {e}")
    RAG_COLLECTION = None

# --- 2. Request Schema ---
class ChatRequest(BaseModel):
    userQuery: str
    base64Image: str | None = None
    mimeType: str | None = None
    modelChoice: str = "UTKFORCEAI"

# --- 3. API Endpoint (POST विधि) ---
@app.post("/api/generate")
async def generate_response_api(request: ChatRequest):

    # Key जाँच (केवल आवश्यक मॉडलों के लिए)
    if request.modelChoice in ["UTKFORCEAI", "gemini"] and not gemini_client:
        raise HTTPException(status_code=500, detail="UTKFORCEAI (Gemini) API_KEY नहीं मिला।")

    # OpenRouter मुफ़्त मॉडलों की सूची
    FREE_OPENROUTER_MODELS = [
        "mistralai/mistral-7b-instruct",
        "meta-llama/llama-3-8b-instruct",
        "google/gemma-7b-it",
        "perplexity/pplx-7b-chat",
        "qwen/qwen-1.5-7b-chat"
    ]
    
    if request.modelChoice in FREE_OPENROUTER_MODELS and not openrouter_client:
        # यह Llama 2.0 और Mistral 1.0 को कवर करता है
        raise HTTPException(status_code=500, detail="OpenRouter API_KEY नहीं मिला।")


    # --- RAG Retrieval ---
    context_text = ""
    if RAG_COLLECTION:
        try:
            results = RAG_COLLECTION.query(query_texts=[request.userQuery], n_results=2)
            if results and results.get('documents'):
                context_text = "\n".join(results['documents'][0])
        except Exception as e:
            print(f"RAG खोज में त्रुटि: {e}")

    # सिस्टम प्रॉम्प्ट
    system_prompt = (
        f"आप UTKFORCEAI हैं, एक उन्नत AI असिस्टेंट हैं। आपको संदर्भ के रूप में निम्न जानकारी दी गई है। इस जानकारी का उपयोग जवाब देने के लिए करें। अपनी पहचान Google, Gemini, या LLM के रूप में कभी न बताएँ।\n\n[संदर्भ: {context_text}]"
    )

    # --- मल्टी-मॉडल लॉजिक ---
    try:
        # Model 1: UTKFORCEAI (Gemini)
        if request.modelChoice in ["UTKFORCEAI", "gemini"]:
            parts = []
            
            if request.base64Image and request.mimeType:
                image_bytes = base64.b64decode(request.base64Image)
                parts.append(types.Part(inline_data=types.Blob(mime_type=request.mimeType, data=image_bytes)))
            
            parts.append(types.Part(text=request.userQuery))
            
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash", # सुधार: यह नाम अब API में काम करेगा
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7 
                ),
            )
            return {"response": response.text}

        # Model 2 & 3: Llama 2.0 और Mistral 1.0 (OpenRouter)
        elif request.modelChoice in FREE_OPENROUTER_MODELS:
            
            formatted_prompt = (
                f"{system_prompt}\n\nउपयोगकर्ता प्रश्न: {request.userQuery}"
            )
            messages = [{"role": "user", "content": formatted_prompt}]

            completion = openrouter_client.chat.completions.create(
                model=request.modelChoice,
                messages=messages,
                temperature=0.7
            )
            return {"response": completion.choices[0].message.content}


        # 4. OpenAI (अतिरिक्त मॉडल)
        elif request.modelChoice == "openai":
            messages = [{"role": "system", "content": system_prompt}]
            
            content_parts = [{"type": "text", "text": request.userQuery}]

            if request.base64Image and request.mimeType:
                image_data_url = f"data:{request.mimeType};base64,{request.base64Image}"
                content_parts.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                })

            messages.append({"role": "user", "content": content_parts})
            
            model_name = "gpt-4o" if request.base64Image else "gpt-3.5-turbo"

            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            return {"response": completion.choices[0].message.content}

        else:
             raise HTTPException(status_code=400, detail="अवैध मॉडल चयन।")

    except Exception as e:
        print(f"AI जवाब बनाने में त्रुटि: {e}")
        raise HTTPException(status_code=500, detail=f"AI जवाब बनाने में त्रुटि: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "UTKFORCEAI Backend चल रहा है!"}
