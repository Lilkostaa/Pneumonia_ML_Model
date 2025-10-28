from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from keras.preprocessing import image
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import base64

# Import Grad-CAM utilities
from gradcam_utils import generate_gradcam_visualization

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
GRADCAM_FOLDER = 'gradcam_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*60)
print("PNEUMONIA DETECTION API - Starting")
print("="*60)

# Rebuild model architecture
print("\n[1/2] Rebuilding model architecture...")
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers[:-4]:
    layer.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Load trained weights
print("[2/2] Loading trained weights...")
try:
    model.load_weights('best_pneumonia_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("\n" + "="*60)
print("🚀 Flask API Ready with Grad-CAM!")
print("="*60 + "\n")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            return None, "URL does not point to an image"
        
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_url_image.jpg')
        img.save(temp_path, 'JPEG')
        
        return temp_path, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def predict_image_with_gradcam(img_path, enable_gradcam=True):
    try:
        # Load and preprocess
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Prediction
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            result = "PNEUMONIA"
            probability = confidence * 100
        else:
            result = "NORMAL"
            probability = (1 - confidence) * 100
        
        result_dict = {
            'diagnosis': result,
            'confidence': round(probability, 2),
            'raw_score': round(confidence, 4),
            'status': 'success'
        }
        
        # Generate Grad-CAM
        if enable_gradcam:
            try:
                gradcam_path = os.path.join(GRADCAM_FOLDER, 'latest_gradcam.png')
                superimposed_img, heatmap = generate_gradcam_visualization(
                    model, img_path, img_array, save_path=gradcam_path
                )
                
                # Convert to base64
                with open(gradcam_path, 'rb') as f:
                    gradcam_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                result_dict['gradcam'] = gradcam_base64
                result_dict['gradcam_available'] = True
            except Exception as e:
                print(f"Grad-CAM error: {e}")
                result_dict['gradcam_available'] = False
        else:
            result_dict['gradcam_available'] = False
        
        return result_dict
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    enable_gradcam = request.args.get('gradcam', 'true').lower() == 'true'
    
    # Handle URL
    if request.is_json or 'url' in request.form:
        data = request.get_json() if request.is_json else request.form
        image_url = data.get('url', '').strip()
        
        if not image_url:
            return jsonify({'error': 'No URL provided'}), 400
        
        temp_path, error = download_image_from_url(image_url)
        
        if error:
            return jsonify({'error': error}), 400
        
        try:
            result = predict_image_with_gradcam(temp_path, enable_gradcam)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result.get('status') == 'success':
                result['source'] = 'url'
                return jsonify(result), 200
            else:
                return jsonify({'error': result.get('message')}), 500
        
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500
    
    # Handle file upload
    elif 'file' in request.files:
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                result = predict_image_with_gradcam(filepath, enable_gradcam)
                
                os.remove(filepath)
                
                if result.get('status') == 'success':
                    result['source'] = 'upload'
                    return jsonify(result), 200
                else:
                    return jsonify({'error': result.get('message')}), 500
            
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    else:
        return jsonify({'error': 'No file or URL provided'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'gradcam_enabled': True,
        'api_version': '2.0'
    }), 200

if __name__ == '__main__':
    print("\n💡 Access: http://localhost:5000")
    print("💡 Features: Upload, URL, Grad-CAM")
    print("💡 Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
