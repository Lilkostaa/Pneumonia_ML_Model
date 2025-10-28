import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    """
    Generate Grad-CAM heatmap - SIMPLIFIED VERSION
    """
    
    # Get the VGG16 base model and the last conv layer
    vgg16_model = model.layers[0]
    last_conv_layer = vgg16_model.get_layer(last_conv_layer_name)
    
    # Create a new model that:
    # - Takes the same input as the original model
    # - Outputs the activations of the last conv layer AND the final predictions
    last_conv_layer_model = keras.Model(vgg16_model.inputs, last_conv_layer.output)
    
    # Create a separate model for the classifier part
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    classifier_model = keras.Model(classifier_input, x)
    
    # Copy weights from the original model to classifier model
    for i, layer in enumerate(classifier_model.layers[1:], start=1):  # Skip input layer
        original_layer = model.layers[i]
        if len(layer.get_weights()) > 0:
            layer.set_weights(original_layer.get_weights())
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Get conv outputs
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        
        # Get predictions
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by importance
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    return heatmap

def apply_gradcam_to_image(img_path, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image
    """
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return superimposed, heatmap_resized

def generate_gradcam_visualization(model, img_path, img_array, save_path='gradcam_result.png'):
    """
    Complete Grad-CAM visualization
    """
    
    print("  → Generating activation heatmap...")
    heatmap = make_gradcam_heatmap(img_array, model)
    
    print("  → Applying heatmap to image...")
    superimposed_img, heatmap_resized = apply_gradcam_to_image(img_path, heatmap, alpha=0.4)
    
    print("  → Creating visualization...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original
    ax1 = plt.subplot(1, 3, 1)
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title('Original Chest X-Ray', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Activation Heatmap', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
    ax2.axis('off')
    
    # Overlay
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(superimposed_img)
    ax3.set_title('Grad-CAM Visualization', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    fig.suptitle('Grad-CAM: AI Focus Areas for Pneumonia Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to: {save_path}")
    plt.close()
    
    return superimposed_img, heatmap

def create_side_by_side_comparison(model, pneumonia_img_path, normal_img_path, 
                                   pneumonia_array, normal_array, save_path='gradcam_comparison.png'):
    """
    Side-by-side comparison
    """
    
    print("  → Generating heatmaps for comparison...")
    
    heatmap_pneumonia = make_gradcam_heatmap(pneumonia_array, model)
    heatmap_normal = make_gradcam_heatmap(normal_array, model)
    
    overlay_pneumonia, _ = apply_gradcam_to_image(pneumonia_img_path, heatmap_pneumonia, alpha=0.5)
    overlay_normal, _ = apply_gradcam_to_image(normal_img_path, heatmap_normal, alpha=0.5)
    
    print("  → Creating comparison figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Pneumonia row
    img_p = cv2.imread(pneumonia_img_path)
    img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_p, cmap='gray')
    axes[0, 0].set_title('PNEUMONIA - Original', fontsize=12, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(heatmap_pneumonia, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('PNEUMONIA - Heatmap', fontsize=12, fontweight='bold', color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(overlay_pneumonia)
    axes[0, 2].set_title('PNEUMONIA - Overlay', fontsize=12, fontweight='bold', color='red')
    axes[0, 2].axis('off')
    
    # Normal row
    img_n = cv2.imread(normal_img_path)
    img_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(img_n, cmap='gray')
    axes[1, 0].set_title('NORMAL - Original', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(heatmap_normal, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title('NORMAL - Heatmap', fontsize=12, fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(overlay_normal)
    axes[1, 2].set_title('NORMAL - Overlay', fontsize=12, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    fig.suptitle('Grad-CAM Comparison: Pneumonia vs Normal X-Rays', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Comparison saved to: {save_path}")
    plt.close()
