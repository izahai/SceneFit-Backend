from fastapi import FastAPI, UploadFile, File, Form
import time

app = FastAPI(title="SceneFit Mock API")

@app.post("/api/v1/retrieval/all-methods")
async def all_methods(
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    # Simulate heavy processing
    time.sleep(10)

    # Fixed mock response
    return {
        "image-edit": [
            {
                "name": "name1",
                "score": 0.24
            }
        ],
        "vlm": [],
        "clip": [],
        "aes": []
    }
