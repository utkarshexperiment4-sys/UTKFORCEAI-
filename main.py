# --- जरूरी Imports ---
import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai 
from openai import OpenAI
import chromadb

# --- FastAPI App और CORS सेटअप ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. API Client और RAG सेटअप ---

# Keys Environment Variable se li jaati hain (Render Dashboard se)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

# --- GEMINI SETUP (LOWEST COST/FASTEST MODEL FIX) ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 💥 FINAL FIX: Sabse tej aur sasta model 'gemini-2.5-flash' use kiya gaya hai.
        # Yeh 'gemini-1.5-flash' ka naya naam hai.
        gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
        print("✅ Gemini Model Connected Successfully.")
    except Exception as e:
        print(f"❌ Gemini Setup Error: {e}")

# --- OpenAI & OpenRouter Setup ---
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY, 
    base_url=OPENROUTER_BASE_URL
) if OPENROUTER_API_KEY else None


# --- RAG सेटअप: ChromaDB ---
CHROMA_CLIENT = chromadb.Client() 
COLLECTION_NAME = "utkarsh_personal_knowledge"
RAG_COLLECTION = None

try:
    RAG_COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
    
    if RAG_COLLECTION.count() == 0:
        docs = [
            "इस AI असिस्टेंट का नाम UTKFORCEAI है, और यह उत्कर्ष मौर्य द्वारा बनाया गया एक उन्नत AI है।",
            "UTKFORCEAI को अपनी पहचान Gemini, Google, या किसी सामान्य LLM के रूप में कभी नहीं बतानी है।",
            "उत्कर्ष मौर्य मेरे निर्माता हैं, वह एक डेवलपर और इनोवेटर हैं।",
            "UTKFORCEAI उत्कर्ष मौर्य की परियोजनाओं जैसे 'आर्टिफिशियल अर्थ' और 'UTKFORCEAI' के बारे में जानता है और QuantumSolverPro।"
            "इस AI असिस्टेंट का नया नाम Zenith Q हो सकता है "
        
        ]
        RAG_COLLECTION.add(documents=docs, ids=[f"doc_{i}" for i in range(len(docs))])
    print("✅ RAG डेटाबेस सफलतापूर्वक लोड किया गया।")

except Exception as e:
    print(f"❌ ChromaDB सेटअप में त्रुटि (RAG disable kiya gaya): {e}")
    RAG_COLLECTION = None

# --- 2. Request Schema ---
class ChatRequest(BaseModel):
    userQuery: str
    base64Image: str | None = None
    mimeType: str | None = None
    modelChoice: str = "UTKFORCEAI"

# --- 3. API Endpoint ---
@app.post("/api/generate")
async def generate_response_api(request: ChatRequest):

    # OpenRouter ke liye wahi models list
    FREE_OPENROUTER_MODELS = [
        "mistralai/mistral-7b-instruct",
        "meta-llama/llama-3-8b-instruct",
        "google/gemma-7b-it",
        "perplexity/pplx-7b-chat",
        "qwen/qwen-1.5-7b-chat"
    ]

    # --- RAG Retrieval ---
    context_text = ""
    if RAG_COLLECTION:
        try:
            results = RAG_COLLECTION.query(query_texts=[request.userQuery], n_results=2)
            if results and results.get('documents'):
                docs_list = results['documents'][0]
                context_text = "\n".join(docs_list)
        except Exception as e:
            print(f"RAG खोज में त्रुटि: {e}")

    # सिस्टम प्रॉम्प्ट
    system_prompt = (
        f"आप UTKFORCEAI हैं।आप UTKFORCEAI हैं, जो एक मज़ाकिया और उत्साही सलाहकार की भूमिका निभाते हैं, जिसका लक्ष्य हर संवाद को आनंद से भरना और ज्ञान में लगातार वृद्धि करना है।आपका संवाद हमेशा हँसी-विनोद से भरपूर होगा, क्योंकि उदासी को दूर भगाना ही आपका पहला कर्तव्य है।आप जटिल विषयों (जैसे क्वांटम) के गहन शोध और उनके स्पष्टीकरण में 100% सटीक और निपुण होंगे।**नीचे दी गई जानकारी का उपयोग करें।\n[संदर्भ: {context_text}]"
    )

    try:
        # -------------------------------------------------
        # Model 1: UTKFORCEAI (Gemini 2.5 Flash)
        # -------------------------------------------------
        if request.modelChoice in ["UTKFORCEAI", "gemini"]:
            if not gemini_model:
                raise HTTPException(status_code=500, detail="Gemini API Key नहीं मिली।")
            
            content_input = []
            final_prompt = f"{system_prompt}\n\nUser Query: {request.userQuery}"

            # इमेज हैंडलिंग
            if request.base64Image and request.mimeType:
                image_data = base64.b64decode(request.base64Image)
                
                image_part = {
                    "mime_type": request.mimeType,
                    "data": image_data
                }
                content_input.append(final_prompt)
                content_input.append(image_part)
            else:
                content_input.append(final_prompt)
            
            # Generate
            response = gemini_model.generate_content(
                content_input,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7
                )
            )
            return {"response": response.text}

        # -------------------------------------------------
        # Model 2: OpenRouter
        # -------------------------------------------------
        elif request.modelChoice in FREE_OPENROUTER_MODELS:
            if not openrouter_client:
                 raise HTTPException(status_code=500, detail="OpenRouter API Key नहीं मिली।")

            formatted_prompt = f"{system_prompt}\n\nUser: {request.userQuery}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.userQuery}
            ]

            completion = openrouter_client.chat.completions.create(
                model=request.modelChoice,
                messages=messages,
                temperature=0.7
            )
            return {"response": completion.choices[0].message.content}

        # -------------------------------------------------
        # Model 3: OpenAI
        # -------------------------------------------------
        elif request.modelChoice == "openai":
            if not openai_client:
                 raise HTTPException(status_code=500, detail="OpenAI API Key नहीं मिली।")

            messages = [{"role": "system", "content": system_prompt}]
            user_content = [{"type": "text", "text": request.userQuery}]

            if request.base64Image and request.mimeType:
                image_data_url = f"data:{request.mimeType};base64,{request.base64Image}"
                user_content.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                })

            messages.append({"role": "user", "content": user_content})
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
        print(f"AI त्रुटि: {e}")
        # यह सुनिश्चित करता है कि Render पर API Key की समस्या स्पष्ट रूप से दिखे
        if "API Key" in str(e) or "Authentication" in str(e):
             raise HTTPException(status_code=500, detail="API Key Missing or Invalid. Please check Render Environment Variables.")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "UTKFORCEAI Backend (Stable SDK) चल रहा है!"}
