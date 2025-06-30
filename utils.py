import os
import csv
import bcrypt
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load the EfficientNet model
def load_cnn_model():
    model_path = os.path.join("models", "best_model.h5")
    return load_model(model_path)

# Authenticate user
def authenticate_user(username, password):
    credentials_path = "user_data/credentials.pkl"
    if not os.path.exists(credentials_path):
        return False
    user_db = joblib.load(credentials_path)
    if username in user_db:
        return bcrypt.checkpw(password.encode(), user_db[username])
    return False

# Register a new user
def register_user(username, password):
    credentials_path = "user_data/credentials.pkl"
    if os.path.exists(credentials_path):
        user_db = joblib.load(credentials_path)
    else:
        user_db = {}
    if username in user_db:
        return False  # User already exists
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user_db[username] = hashed_pw
    joblib.dump(user_db, credentials_path)
    return True

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha if present
    return np.expand_dims(img_array, axis=0)

# Predict tumor type
def classify_image(model, image):
    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]  # Maintain this order
    processed = preprocess_image(image)
    predictions = model.predict(processed)[0]
    confidence = float(np.max(predictions))
    class_idx = int(np.argmax(predictions))
    return labels[class_idx], confidence, dict(zip(labels, map(float, predictions)))

# Suggest biopsy (example logic: suggest if not "No Tumor")
def suggest_biopsy(predicted_label):
    if predicted_label == "No Tumor":
        return False
    return True

# Save result to CSV
def save_result(username, predicted_label, confidence):
    file_path = "user_data/results.csv"
    os.makedirs("user_data", exist_ok=True)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, predicted_label, f"{confidence:.2f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

# Generate a PDF report
def generate_pdf_report(username, label, confidence, biopsy_needed, image):
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{username}.pdf"
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, f"CerebroLens Report - {username}")

    # Content
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 130, f"Tumor Prediction: {label}")
    c.drawString(50, height - 150, f"Confidence: {confidence:.2f}")
    c.drawString(50, height - 170, f"Biopsy Suggested: {'Yes' if biopsy_needed else 'No'}")

    # Save uploaded image to disk temporarily
    img_path = f"reports/{username}_uploaded.png"
    image.save(img_path)
    c.drawImage(img_path, 50, height - 450, width=200, preserveAspectRatio=True)

    c.save()
    os.remove(img_path)
    return report_path
