import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("PNEUMONIA DETECTION - FULL TEST SET EVALUATION")
print("="*70)

# Download dataset
print("\n[1/4] Loading dataset...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
test_dir = path + '/chest_xray/test'
print(f"✅ Test directory: {test_dir}")

# Rebuild model architecture
print("\n[2/4] Rebuilding model architecture...")
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
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# Load trained weights
print("Loading trained weights...")
model.load_weights('best_pneumonia_model.h5')
print("✅ Model ready!")

# Prepare test data
print("\n[3/4] Preparing test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"\n📊 Dataset Info:")
print(f"   Total test images: {test_generator.samples}")
print(f"   Classes: {test_generator.class_indices}")
print(f"   Batch size: 32")

# Evaluate on test set
print("\n[4/4] Evaluating model on test set...")
print("This may take a few minutes...\n")

test_results = model.evaluate(test_generator, verbose=1)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_precision = test_results[2]
test_recall = test_results[3]
test_auc = test_results[4]

print("\n" + "="*70)
print("TEST SET EVALUATION RESULTS")
print("="*70)
print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Test Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"Test AUC:       {test_auc:.4f}")
print(f"Test Loss:      {test_loss:.4f}")
print("="*70)

# Generate predictions for confusion matrix
print("\n📈 Generating detailed predictions...")
predictions = model.predict(test_generator, verbose=1)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16})
plt.title('Confusion Matrix - Test Set (624 images)', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.show()

# Classification Report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, 
                          target_names=['NORMAL', 'PNEUMONIA'],
                          digits=4))

# Calculate additional medical metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("\n" + "="*70)
print("MEDICAL METRICS (Important for Healthcare)")
print("="*70)
print(f"Sensitivity (Recall):        {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  → Ability to detect pneumonia cases")
print(f"\nSpecificity:                 {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  → Ability to identify normal cases")
print(f"\nFalse Positive Rate:         {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
print(f"  → Healthy patients wrongly diagnosed")
print(f"\nFalse Negative Rate:         {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
print(f"  → Sick patients missed (CRITICAL!)")
print(f"\nF1 Score:                    {f1_score:.4f}")
print(f"  → Balance between precision and recall")
print("="*70)

# Summary statistics
print("\n" + "="*70)
print("CONFUSION MATRIX BREAKDOWN")
print("="*70)
print(f"True Negatives (TN):   {tn:4d} - Correctly identified NORMAL")
print(f"False Positives (FP):  {fp:4d} - NORMAL wrongly classified as PNEUMONIA")
print(f"False Negatives (FN):  {fn:4d} - PNEUMONIA wrongly classified as NORMAL ⚠️")
print(f"True Positives (TP):   {tp:4d} - Correctly identified PNEUMONIA")
print("="*70)

# Performance visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [test_accuracy, test_precision, test_recall, specificity, f1_score]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']

axes[0].bar(metrics_names, metrics_values, color=colors, alpha=0.8)
axes[0].set_ylim([0, 1.0])
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Pie chart of predictions
correct = tn + tp
incorrect = fp + fn
axes[1].pie([correct, incorrect], 
            labels=[f'Correct\n{correct} images', f'Incorrect\n{incorrect} images'],
            colors=['#4CAF50', '#F44336'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Overall Prediction Accuracy', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: evaluation_metrics.png")
plt.show()

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print("\n📁 Generated files:")
print("  1. confusion_matrix.png - Visual confusion matrix")
print("  2. evaluation_metrics.png - Performance charts")
print("\n💡 These metrics are essential for:")
print("  - Understanding model performance on unseen data")
print("  - Identifying strengths and weaknesses")
print("  - Medical validation and publication")
print("="*70)
