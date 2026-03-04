import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic

app = FastAPI(title="MARS API")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ReasonRequest(BaseModel):
    query: str

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
