from fastapi import FastAPI
from model import load_model, predict_batch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Baseline Image Quality API")

# allow all origins for testing/demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# load dummy model (found in model.py)
model = load_model()

# define the endpoint for predictions
@app.post("/predict")
def predict(request: dict):
    images = request.get("images", [])
    batch_outputs = predict_batch(model, images)

    results = [] 
    for img, output in zip(images, batch_outputs):
        results.append({
            "image_id": img.get("image_id"),
            "scores": output["scores"],
            "decisions": output["decisions"]
        })

    # format the return to match output JSON response schema
    return {
        "request_id": request.get("request_id", ""),
        "results": results
    }