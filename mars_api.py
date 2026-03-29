import os
from fastapi import FastAPI, HTTPException, Header, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

import anthropic

app = FastAPI(title="MARS API")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ReasonRequest(BaseModel):
    query: str

class NWUPayload(BaseModel):
    nwu_version: str
    event_type: str
    source: str
    target: str
    auth_token: str
    payload: Dict[str, Any]
    timestamp: str

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "MARS"}

@app.post("/reason")
async def reason(request: ReasonRequest, x_api_key: str = Header(None)):
    if x_api_key != os.getenv("MARS_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid key")

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": request.query}]
    )
    return {"reasoning": response.content[0].text, "success": True}

@app.post("/nwu-listener")
async def nwu_listener(data: NWUPayload):
    # Verify Internal Agent Token for NWU Protocol
    expected_token = os.getenv("INTERNAL_AGENT_TOKEN")
    if not expected_token or data.auth_token != expected_token:
        print(f"⚠️ NWU: Unauthorized access attempt from {data.source}")
        raise HTTPException(status_code=401, detail="Unauthorized NWU payload")

    print(f"📥 NWU: Received {data.event_type} from {data.source}")
    
    if data.event_type == "opportunity_detected":
        opportunity = data.payload
        print(f"🚀 MARS: Processing opportunity {opportunity.get('id')} - {opportunity.get('description')}")
        # Logic for MARS to begin agentic deal sourcing/processing
        return {
            "status": "acknowledged",
            "nwu_receipt": datetime.utcnow().isoformat() + "Z",
            "action": "queued_for_processing"
        }
    
    return {"status": "ignored", "reason": "unsupported_event_type"}