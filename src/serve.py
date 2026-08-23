import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

from model import build_model

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    model_name = config["output"].get(
        "model_name",
        config["output"].get("checkpoint_filename", "best_model.pth"),
    )
    checkpoint_path = checkpoint_dir / model_name

    if checkpoint_path.exists():
        model = build_model(
            num_classes=config["model"]["num_classes"],
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Successfully loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")

    yield


app = FastAPI(title="Image Classifier API", lifespan=lifespan)


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0).tolist()

        class_probs = dict(zip(CIFAR10_CLASSES, probs))
        top_class = CIFAR10_CLASSES[probs.index(max(probs))]

        return {
            "predicted_class": top_class,
            "probabilities": class_probs,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process image: {str(e)}"
        )

# curl.exe -X POST "http://localhost:8000/predict" -F "file=@C:\Users\Alan Koshy\Downloads\OIP-999065008.jpg"

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
