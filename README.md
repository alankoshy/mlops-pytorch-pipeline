# PyTorch MLOps Pipeline

An end-to-end MLOps pipeline for training, checkpointing, and serving image classification models (ResNet-18 on CIFAR-10) using PyTorch, FastAPI, Docker, and Kubernetes.

---

## 📁 Repository Structure

```
mlops-pytorch-pipeline/
├── configs/
│   └── training_config.yaml      # Model, training, data, and output hyperparameters
├── docker/
│   ├── Dockerfile.train          # Container setup for model training
│   └── Dockerfile.serve          # Container setup for FastAPI model serving
├── k8s/
│   ├── namespace.yaml            # Defines 'ml-training' namespace
│   ├── configmap.yaml            # ConfigMap for training configuration
│   ├── training-job.yaml         # K8s batch Job with GPU scheduling
│   ├── serving-deployment.yaml   # 2-replica Deployment with probes
│   ├── serving-service.yaml      # ClusterIP Service for serving endpoint
│   └── hpa.yaml                  # Horizontal Pod Autoscaler (1-4 replicas)
├── requirements/
│   ├── train.txt                 # Dependencies for model training
│   └── serve.txt                 # Dependencies for inference service
├── src/
│   ├── dataset.py                # CIFAR-10 data loaders & transformations
│   ├── model.py                  # ResNet-18 architecture builder
│   ├── train.py                  # Training pipeline with early stopping
│   └── serve.py                  # FastAPI server with /health and /predict
├── tests/
│   └── test_model.py             # Model unit test suite
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

The pipeline behavior is controlled via [`configs/training_config.yaml`](configs/training_config.yaml):

```yaml
data:
  dataset: "CIFAR10"
  data_dir: "./data"
  num_workers: 8

model:
  architecture: "resnet18"
  num_classes: 10
  pretrained: true

training:
  batch_size: 128
  epochs: 20
  learning_rate: 0.01
  device: "cuda"
  early_stopping_patience: 3

output:
  checkpoint_dir: "./checkpoints"
  model_name: "best_model.pth"
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, CPU fallback supported)
- Docker & Kubernetes cluster (`kubectl`)

### 2. Local Environment Setup

Clone the repository and install dependencies:

```bash
# Install training dependencies
pip install -r requirements/train.txt

# Install serving dependencies
pip install -r requirements/serve.txt
```

---

## 🏋️ Model Training

Run the training pipeline locally:

```bash
python -m src.train
```

Key training features:
- **Pretrained Weights**: Fine-tunes ResNet-18 initialized with ImageNet weights.
- **Data Augmentations**: Random horizontal flip, random cropping (32x32, padding 4), and CIFAR-10 normalization.
- **Early Stopping**: Monitors validation loss (`patience=3`, `min_delta=0.001`).
- **Structured Logging**: Outputs JSON-formatted progress logs per epoch.
- **Checkpointing**: Automatically saves the model with minimal validation loss to `./checkpoints/best_model.pth`.

---

## 🌐 Model Inference & Serving

Start the FastAPI serving app:

```bash
python -m src.serve
```
*(Runs on `http://0.0.0.0:8000` by default)*

### API Endpoints

#### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

#### 2. Image Prediction
- **Endpoint**: `POST /predict`
- **Body**: Form-data with file field `image`
- **Example cURL**:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -F "image=@/path/to/image.jpg"
  ```
- **Response**:
  ```json
  {
    "predicted_class": "cat",
    "probabilities": {
      "airplane": 0.01,
      "automobile": 0.00,
      "bird": 0.02,
      "cat": 0.85,
      "deer": 0.01,
      "dog": 0.08,
      "frog": 0.01,
      "horse": 0.00,
      "ship": 0.00,
      "truck": 0.02
    }
  }
  ```

---

## 🐳 Containerization with Docker

### Build Images

```bash
# Build training container
docker build -t mlops-pytorch-pipeline-train:latest -f docker/Dockerfile.train .

# Build serving container
docker build -t mlops-pytorch-pipeline-serve:latest -f docker/Dockerfile.serve .
```

### Run Containers

```bash
# Run training
docker run --rm -v $(pwd)/checkpoints:/app/checkpoints mlops-pytorch-pipeline-train:latest

# Run inference service
docker run -d -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints --name model-serving mlops-pytorch-pipeline-serve:latest
```

---

## ☸️ Kubernetes Deployment

Deploy the entire pipeline to Kubernetes under the `ml-training` namespace:

### 1. Setup Namespace & ConfigMap

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
```

### 2. Run Training Job

```bash
kubectl apply -f k8s/training-job.yaml
```
*Specifies GPU node selector (`accelerator: nvidia-gpu`) and mounts persistent host paths for data and model checkpoints.*

### 3. Deploy Serving & Autoscaling

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4. Monitor Deployment

```bash
# Check pod status
kubectl get pods -n ml-training

# Check HPA status
kubectl get hpa -n ml-training

# Check logs
kubectl logs -l app=model-serving -n ml-training
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
