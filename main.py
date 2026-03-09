from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import analyze
import uvicorn
import uvicorn

app = FastAPI(
    title="Blindness Detection & DR Progression API",
    description="AI-powered pipeline for Diabetic Retinopathy detection, explainability, and progression.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the MedVision AI Backend"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
