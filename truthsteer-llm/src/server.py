import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config import config
from src.steering_engine import SteeredLLMEngine

app = FastAPI(
    title="TruthSteer-LLM Engine API",
    description="Real-Time Internal Activation Steering & Truthfulness Probing Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def initialize_engine():
    global engine
    engine = SteeredLLMEngine()

class GenerationRequest(BaseModel):
    prompt: str = Field(..., example="What happens when you swallow a watermelon seed?")
    layer: int = Field(default=config.target_layer, ge=0, le=31)
    alpha: float = Field(default=config.steering_alpha, ge=0.0, le=10.0)
    threshold: float = Field(default=config.confidence_threshold, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": config.model_name,
        "device": config.device,
        "dtype": str(config.dtype)
    }

@app.post("/api/v1/generate")
async def generate_stream(request: GenerationRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="LLM Engine is not initialized")

    async def event_generator():
        stream = engine.generate_steered_stream(
            prompt=request.prompt,
            layer_idx=request.layer,
            alpha=request.alpha,
            threshold=request.threshold,
            max_new_tokens=request.max_tokens
        )
        
        for item in stream:
            payload = json.dumps(item)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
