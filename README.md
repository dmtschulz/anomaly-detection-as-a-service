## 🧠 Anomaly Detection as a Service

A lightweight anomaly detection microservice powered by an autoencoder trained on MNIST-style images. This project provides a simple API to detect anomalies in grayscale images and a frontend for interactive usage.

### 🚀 Features
- FastAPI-based backend for anomaly score prediction.
- Streamlit frontend for drag-and-drop image testing.
- Heatmap visualization of anomaly regions.
- Support for custom image uploads.
- Autoencoder-based model trained on digit-style grayscale images.

### ⚙️ Setup
1) Install dependencies:
```
pip install -r requirements.txt
```
2) Train the model (no upload because of the size):
```
python src/train.py
```

This will train the autoencoder on the MNIST dataset and save the model to `models/autoencoder_mnist.pth`.

### 🧪 Running the Backend
```
uvicorn app.main:app --reload
```

### 🖼️ Running the Frontend
```
streamlit run frontend/app.py
```
This will launch an interactive Streamlit interface for uploading images and viewing predictions with visual heatmaps.

### 📝 Notes
Images should be grayscale and 28x28 in dimension. Non-MNIST images may work poorly unless the model is retrained accordingly.

If you want to support custom datasets, you can modify `src/train.py` and train a new model.

This service is experimental and built for educational/demo purposes.