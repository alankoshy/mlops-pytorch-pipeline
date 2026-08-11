import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    config_path = os.path.join("configs", "training_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cfg_train = config["training"]
        cfg_model = config["model"]
        checkpoint_path = os.path.join(
            cfg_train["checkpoint_dir"], cfg_train["checkpoint_filename"]
        )

        if os.path.exists(checkpoint_path):
            model = build_model(
                num_classes=cfg_model["num_classes"],
                pretrained=False,
                in_channels=3,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
