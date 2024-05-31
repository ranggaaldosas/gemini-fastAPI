from fastapi import APIRouter, File, UploadFile

from app.services.ai_processing import summarize_text
from app.services.file_handling import extract_text_from_pdf, save_upload_file

router = APIRouter()


@router.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_location = await save_upload_file(file)
    extracted_text = extract_text_from_pdf(file_location)
    summary = summarize_text(extracted_text)
    return {"filename": file.filename, "summary": summary}
