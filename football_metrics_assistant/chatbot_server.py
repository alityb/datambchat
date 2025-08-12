#!/usr/bin/env python3
"""
Simple server to serve the chatbot frontend and integrate with the football metrics API.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from football_metrics_assistant.main import app as api_app

# Create the main app
app = FastAPI(title="Football Metrics Assistant Chatbot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API routes from the main app
app.mount("/api", api_app)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chatbot():
    """Serve the chatbot frontend."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "chatbot.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chatbot frontend not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Football Metrics Assistant Chatbot"}

if __name__ == "__main__":
    print("🚀 Starting Football Metrics Assistant Chatbot Server...")
    print("📱 Chatbot available at: http://localhost:8080")
    print("🔌 API available at: http://localhost:8080/api")
    print("📊 Health check at: http://localhost:8080/health")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "chatbot_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 