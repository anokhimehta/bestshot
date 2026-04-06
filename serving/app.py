from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from serving.model import load_model, predict_batch  # local
except ModuleNotFoundError:
    from model import load_model, predict_batch          # docker

    
app = FastAPI(title="Baseline BestShot API")

# allow all origins for testing/demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# load model at startup
model = load_model()

@app.post("/predict")
def predict(request: dict):
    images = request.get("images", [])

    # pass the full image dict
    batch_outputs = predict_batch(model, images) # predict_batch handles path extraction

    results = []
    for img, output in zip(images, batch_outputs):
        results.append({
            "image_id": img.get("image_id"),
            "scores": output["scores"],
            "decisions": output["decisions"]
        })

    return {
        "request_id": request.get("request_id", ""),
        "results": results
    }