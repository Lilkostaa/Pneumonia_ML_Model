import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# Load model
model = keras.models.load_model('best_pneumonia_model.h5')

def predict_pneumonia(image):
    """Predict pneumonia from chest X-ray"""
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        diagnosis = "⚠️ PNEUMONIA DETECTED"
        confidence = prediction * 100
        color = "red"
    else:
        diagnosis = "✅ NORMAL CHEST X-RAY"
        confidence = (1 - prediction) * 100
        color = "green"
    
    return f"**{diagnosis}**\n\nConfidence: {confidence:.2f}%\nRaw Score: {prediction:.4f}"

# Create interface
demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Markdown(label="Diagnosis"),
    title="🫁 Pneumonia AI Diagnostics",
    description="AI-powered chest X-ray analysis using VGG16 Transfer Learning\n\n⚠️ For educational purposes only - Not for clinical use",
    examples=[
        ["examples/pneumonia_example.jpg"],
        ["examples/normal_example.jpg"]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
