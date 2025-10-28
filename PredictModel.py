import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*60)
print("PNEUMONIA DETECTION - PREDICTION SCRIPT")
print("="*60)

print("\n[1/3] Rebuilding model architecture...")

# Rebuild the EXACT same architecture used in training
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze early layers (same as training)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Construct the complete model (same as training)
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

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model architecture created")

print("\n[2/3] Loading trained weights...")
try:
    # Load only the weights from the .h5 file
    model.load_weights('best_pneumonia_model.h5')
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("\nMake sure 'best_pneumonia_model.h5' is in the current directory:")
    print(f"Current directory: {os.getcwd()}")
    exit(1)

def predict_pneumonia(img_path):
    """
    Predict if a chest X-ray shows pneumonia or is normal
    
    Args:
        img_path: Path to the X-ray image file
        
    Returns:
        result: "PNEUMONIA" or "NORMAL"
        prob: Confidence percentage
        confidence: Raw probability score
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0]
    
    # Interpret results
    if confidence > 0.5:
        result = "PNEUMONIA"
        prob = confidence * 100
    else:
        result = "NORMAL"
        prob = (1 - confidence) * 100
    
    return result, prob, confidence

def visualize_prediction(img_path):
    """
    Display the X-ray image with prediction results
    """
    result, prob, confidence = predict_pneumonia(img_path)
    
    # Display image
    img = image.load_img(img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Add prediction text
    color = 'red' if result == "PNEUMONIA" else 'green'
    plt.title(f'Prediction: {result}\nConfidence: {prob:.2f}%', 
              fontsize=16, fontweight='bold', color=color)
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"Diagnosis:  {result}")
    print(f"Confidence: {prob:.2f}%")
    print(f"Raw Score:  {confidence:.4f}")
    print(f"{'='*60}\n")
    
    return result, prob

# Example usage
from gradcam_utils import generate_gradcam_visualization

# Example usage
if __name__ == "__main__":
    import kagglehub
    
    print("\n[3/3] Preparing test data...")
    
    # Download dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"✅ Dataset path: {path}\n")
    
    print("="*60)
    print("TESTING MODEL WITH GRAD-CAM")
    print("="*60)
    
    # Test with PNEUMONIA case
    print("\n🔬 Test 1: PNEUMONIA Case\n")
    test_image_1 = path + '/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'
    print(f"Image: {test_image_1}")
    
    # Make prediction
    result1, prob1 = visualize_prediction(test_image_1)
    
    # Generate Grad-CAM
    print("\n🔥 Generating Grad-CAM visualization...")
    
    # Load and preprocess for Grad-CAM
    img = image.load_img(test_image_1, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Import Grad-CAM utilities
    from gradcam_utils import generate_gradcam_visualization
    
    superimposed_img, heatmap = generate_gradcam_visualization(
        model, test_image_1, img_array, save_path='gradcam_pneumonia.png'
    )
    
    # Test with NORMAL case
    print("\n" + "="*60)
    print("\n🔬 Test 2: NORMAL Case\n")
    test_image_2 = path + '/chest_xray/test/NORMAL/IM-0001-0001.jpeg'
    print(f"Image: {test_image_2}")
    
    # Make prediction
    result2, prob2 = visualize_prediction(test_image_2)
    
    # Generate Grad-CAM
    print("\n🔥 Generating Grad-CAM visualization...")
    
    # Load and preprocess for Grad-CAM
    img = image.load_img(test_image_2, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    superimposed_img, heatmap = generate_gradcam_visualization(
        model, test_image_2, img_array, save_path='gradcam_normal.png'
    )
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  - prediction_result.png (last prediction)")
    print("  - gradcam_pneumonia.png (Grad-CAM for pneumonia case)")
    print("  - gradcam_normal.png (Grad-CAM for normal case)")
    print("\n💡 Red areas in Grad-CAM show where the AI focused its attention!")
        # Create side-by-side comparison
    print("\n" + "="*60)
    print("\n📊 Creating comparison visualization...")
    
    from gradcam_utils import create_side_by_side_comparison
    
    # Load both images
    img_p = image.load_img(test_image_1, target_size=(224, 224))
    img_p_array = image.img_to_array(img_p)
    img_p_array = np.expand_dims(img_p_array, axis=0) / 255.0
    
    img_n = image.load_img(test_image_2, target_size=(224, 224))
    img_n_array = image.img_to_array(img_n)
    img_n_array = np.expand_dims(img_n_array, axis=0) / 255.0
    
    create_side_by_side_comparison(
        model, test_image_1, test_image_2,
        img_p_array, img_n_array,
        save_path='gradcam_comparison.png'
    )
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\n📁 Generated files:")
    print("  1. prediction_result.png - Last prediction")
    print("  2. gradcam_pneumonia.png - Grad-CAM for pneumonia")
    print("  3. gradcam_normal.png - Grad-CAM for normal")
    print("  4. gradcam_comparison.png - Side-by-side comparison")
    print("\n🔥 Red areas show where the AI focused attention!")
    print("💡 Notice how pneumonia cases show focused hot spots")
    print("="*60 + "\n")



