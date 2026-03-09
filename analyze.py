import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64

from backend.services.image_service import validate_image, preprocess_image
from backend.services.ml_service import ml_service
from backend.services.llm_service import llm_service
from backend.services.pdf_service import pdf_service

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    diagnosis: dict | None = None
    interpretation: str | None = None
    report_url: str | None = None

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_retinal_image(file: UploadFile = File(...)):
    """Core endpoint that runs the entire ML pipeline"""
    
    file_bytes = await file.read()
    
    # 1. Validation
    if not validate_image(file_bytes):
        raise HTTPException(status_code=400, detail="Invalid image format or corrupted file.")
        
    # 2. Preprocessing
    processed_img = preprocess_image(file_bytes)
    
    # 3. Model Inference (ViT -> U-Net -> Multi-Task -> LSTM -> Grad-CAM)
    diagnosis_data = ml_service.analyze_image(processed_img)
    
    # 4. LLM Interpretation & Recommendation Engine
    interpretation = llm_service.generate_clinical_interpretation(diagnosis_data)
    
    # 5. Generate PDF Report
    report_filename = f"report_{file.filename.split('.')[0]}.pdf"
    pdf_service.generate_report(diagnosis_data, interpretation, report_filename)
    
    return AnalysisResponse(
        success=True,
        message="Retinal scan processed successfully.",
        diagnosis=diagnosis_data,
        interpretation=interpretation,
        report_url=f"/api/v1/download/{report_filename}"
    )

@router.get("/download/{filename}")
async def download_report(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename, media_type='application/pdf', filename=filename)
    raise HTTPException(status_code=404, detail="Report not found")
