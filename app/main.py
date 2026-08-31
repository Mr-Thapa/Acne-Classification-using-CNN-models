from fastapi import FastAPI,UploadFile,File, HTTPException

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
    