from fastapi import FastAPI,UploadFile,File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.inference import predict,Invalid_Image_Exception

app=FastAPI()
@app.get("/health")
def health():
    return {"status":"ok"}

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/predict")
async def predict_image(file:UploadFile=File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )
    image=await file.read()
    try:
        result=predict(image)
    except Invalid_Image_Exception:
        raise HTTPException(
                    status_code=400,
                    detail="Invalid or Corrupted Image uploaded"
            )
    return result
    