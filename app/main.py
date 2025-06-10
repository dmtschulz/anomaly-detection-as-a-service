# app/main.py

from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
import torch
from torchvision import transforms as T
from src.train import load_model, loss_fn  # Load the model and loss function from src/train.py
import io

from fastapi.responses import JSONResponse
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = FastAPI()

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model once
MODEL_PATH = "./models/autoencoder_mnist.pth"
model = load_model(MODEL_PATH)
model.eval()

# Transformations
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor()
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image = transform(image)
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            decoded = model(image)
            score = loss_fn(decoded, image).item()
            diff = (decoded - image).squeeze().cpu().numpy() ** 2

        # Create heatmap image
        fig, ax = plt.subplots()
        ax.imshow(diff, cmap="hot")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Encode to base64
        heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Create reconstructed image
        decoded_img = decoded.squeeze().cpu().numpy()

        # Visualize the decoded image as png
        fig2, ax2 = plt.subplots()
        ax2.imshow(decoded_img, cmap="gray")
        ax2.axis("off")
        buf2 = BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig2)
        buf2.seek(0)

        # Decode to base64
        decoded_b64 = base64.b64encode(buf2.read()).decode("utf-8")

        # Return everything as JSON
        return JSONResponse(content={
            "anomaly_score": score,
            "heatmap": heatmap_b64,
            "decoded_image": decoded_b64
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
