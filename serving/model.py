import random
import time

# gerenating dummy model and predictions for testing purposes (output json response)

def load_model():
    return "dummy_model"

def predict_batch(model, images):
    # fake simulation of compute time (~10ms per image)
    time.sleep(0.01 * len(images))

    results = []
    for img in images:
        scores = {
            "koniq_score": random.uniform(0.5, 1.0),
            "sharpness": random.uniform(0.5, 1.0),
            "exposure": random.uniform(0.5, 1.0),
            "face_quality": random.uniform(0.5, 1.0),
        }

        composite_score = sum(scores.values()) / len(scores)
        scores["composite_score"] = composite_score

        decisions = {
            "quality_label": "high_quality" if composite_score > 0.8 else "low_quality",
            "review_flag": "keep" if composite_score > 0.7 else "review_for_deletion",
            "is_best_shot": composite_score > 0.9,
            "is_burst": False,
            "burst_group_id": None
        }

        results.append({
            "scores": scores,
            "decisions": decisions
        })

    return results