import os
import shutil

import PyPDF2
from fastapi import UploadFile


async def save_upload_file(upload_file: UploadFile) -> str:
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, upload_file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_location


def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text
