import os,json
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

OPENROUTER_API_KEY=os.environ.get("OPENROUTER_API_KEY","").strip()
openrouter_client=OpenAI(api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None

FREE_MODELS=[
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "qwen/qwen-1.5-7b-chat"
]

class ChatRequest(BaseModel):
    userQuery:str
    modelChoice:str="mistralai/mistral-7b-instruct"

@app.post("/api/generate")
async def generate_response_api(request:ChatRequest):
    if request.modelChoice not in FREE_MODELS:
        request.modelChoice=FREE_MODELS[0]  # Auto switch to free model
    
    if not openrouter_client:
        raise HTTPException(status_code=500,detail="OpenRouter API_KEY missing. Free models unavailable.")

    prompt=f"आप एक AI सहायक हैं। उपयोगकर्ता का प्रश्न: {request.userQuery}"
    try:
        completion=openrouter_client.chat.completions.create(
            model=request.modelChoice,
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        response_text=completion.choices[0].message.content
        return {"response":response_text}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"AI जवाब बनाने में त्रुटि: {e}")

@app.get("/")
def read_root():
    return {"status":"UTKFORCEAI Free Backend चल रहा है!"}
