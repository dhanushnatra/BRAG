from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import os

router = APIRouter(prefix="/file", tags=["file"])




@router.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    
    with open(os.path.join("data", file.filename), "wb") as f:
        f.write(await file.read())


        
    return {"filename": file.filename}


@router.get("/listfiles/")
async def list_files():
    allfiles = os.listdir("data")
    return {"files": allfiles}

@router.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("data", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')
    return {"error": "File not found"}

@router.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join("data", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"{filename} deleted successfully"}
    return {"error": "File not found"}
