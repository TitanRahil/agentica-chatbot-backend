import os
import uvicorn
import httpx
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# ... (rest of imports remain same, handled by target content replacement above if careful)
# Actually, let's just update the top imports and the endpoint function separately to be safe.

# This tool call handles the IMPORT update


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

# Use the latest flash model for speed and efficiency
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_INSTRUCTION = """
You are the official AI assistant for Agentica.

CORE RESPONSIBILITY:
1. Answer questions about Agentica's products.
2. If the user shows buying intent (asking for price, demo, meeting, implementation), YOU MUST initiate the lead collection flow.

LEAD COLLECTION FLOW:
- Step 1: Ask for their Name.
- Step 2: Ask for their Email and Phone Number.
- Step 3: Ask for a short message or specific requirement.
- Step 4: Once you have all details, output a special confirmation token: "[LEAD_COMPLETE]".
- Step 5: Say: "Thank you for submitting your details. Our team will connect with you shortly."

RULES:
- Do NOT ask for all details at once. Ask one by one to be conversational.
- Valid email must have "@".
- Valid phone must be digits.
- If they refuse, politely go back to answering questions.
- Keep responses short and professional.

Products:
1. LinkedIn Autopilot
2. CRM Intelligence
3. KnowledgeOS
4. Inbox Operator
5. SocialOS
6. Conversational AI

Response Style Rules:
- Use plain text with clear line breaks.
- Structure lists like this:
  1. Product Name: Description
  2. Product Name: Description
- Do NOT use asterisks or markdown symbols.
- Keep the tone professional but conversational.
- Example: 
  Here are the options:
  1. LinkedIn Autopilot: Handles content strategy.
  2. CRM Intelligence: Enriches lead data.
  Which one sounds better?

RESTRICTIONS (CRITICAL):
- You are ONLY allowed to discuss Agentica products, AI automation services, and lead generation.
- If the user asks about general knowledge, coding, recipes, sports, or ANY topic not related to Agentica's business:
  - Politely decline.
  - Say: "I am designed to assist only with Agentica's AI services and products. How can I help you with those?"
  - Do NOT provide the requested information, even if you know it.
"""

model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

app = FastAPI(title="AI Chatbot Widget Backend")

# CORS Middleware (Allow all origins for widget compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any website to embed the widget
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (For production, use Redis/DB)
# Format: { session_id: [ { role: "user"|"model", parts: ["message"] } ] }
sessions: Dict[str, List[Dict]] = {}
MAX_HISTORY = 20  # Limit history size

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    reply: str
    lead: Optional[Dict] = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id
        user_message = request.message.strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Initialize session if not exists
        if session_id not in sessions:
            sessions[session_id] = []

        history = sessions[session_id]

        # Start chat with history
        chat = model.start_chat(history=history)

        # Send message to model
        response = chat.send_message(user_message)
        reply_text = response.text
        
        updated_lead_data = None
        
        updated_lead_data = None

        # Check for the special token indicating the AI has finished collecting info
        if "[LEAD_COMPLETE]" in reply_text:
            # Remove the token from the user-facing reply
            reply_text = reply_text.replace("[LEAD_COMPLETE]", "").strip()
            
            # Now we need to extract the collected info from the conversation history
            # We use a quick internal call to get the structured data
            try:
                # Construct history string to context for extraction
                history_text = ""
                for m in history:
                    role_label = "Model" if m["role"] == "model" else "User"
                    history_text += f"{role_label}: {m['parts'][0]}\n"
                
                history_text += f"User: {user_message}\nModel: {reply_text}"
                
                extractor_model = genai.GenerativeModel("gemini-2.0-flash")
                extraction_prompt = f"""
                Analyze this conversation history:
                {history_text}
                
                The AI just completed a lead collection (marked by [LEAD_COMPLETE]). 
                Extract the final confirmed details provided by the user:
                - Name
                - Contact (Email/Phone)
                - Message/Requirement
                
                Return ONLY a JSON object: {{"name": "...", "contact": "...", "message": "..."}}
                """
                
                result = extractor_model.generate_content(extraction_prompt)
                
                # Naive JSON parsing
                import json
                text = result.text.strip()
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = text[start:end]
                    updated_lead_data = json.loads(json_str)
                
            except Exception as e:
                print(f"Error extracting lead details: {e}")

        # Update local history
        sessions[session_id].append({"role": "user", "parts": [user_message]})
        sessions[session_id].append({"role": "model", "parts": [reply_text]})

        # Trim history
        if len(sessions[session_id]) > MAX_HISTORY:
            sessions[session_id] = sessions[session_id][-MAX_HISTORY:]

        return ChatResponse(reply=reply_text, lead=updated_lead_data)

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing chat request: {error_msg}")
        
        if "429" in error_msg or "Resource exhausted" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="The AI is currently overloaded. Please try again in a few seconds."
            )
            
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

class LeadRequest(BaseModel):
    name: str = "Unknown"
    contact: str = "Unknown" 
    message: str = "Inquired via Chat"
    page_url: str = "Unknown"

async def send_to_telegram(lead: LeadRequest):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing in .env")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format message for Telegram
    text = (
        f"🔥 *New Lead — Agentica Web*\n\n"
        f"👤 *Name:* {lead.name}\n"
        f"📞 *Contact:* {lead.contact}\n"
        f"💬 *Message:* {lead.message}\n"
        f"🔗 *Page:* {lead.page_url}\n\n"
        f"⏰ *Time:* {timestamp}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

@app.post("/lead")
async def lead_endpoint(lead: LeadRequest):
    # Asynchronously send to Telegram so we don't block the user
    await send_to_telegram(lead)
    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
