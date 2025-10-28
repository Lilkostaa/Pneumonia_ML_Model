# Pneumonia Detection from Chest X-Ray Images

A deep learning project that uses Convolutional Neural Networks (CNN) with Transfer Learning to detect pneumonia from chest X-ray images. The model is built using TensorFlow/Keras and trained on the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## 🎯 Overview

This project aims to automatically detect pneumonia in chest X-ray images using deep learning techniques. The model employs Transfer Learning with VGG16 as the base architecture, achieving high accuracy in distinguishing between normal and pneumonia-affected lungs.

**Key Features:**
- Transfer Learning with pre-trained VGG16
- Data augmentation for improved generalization
- Multiple evaluation metrics (Accuracy, Precision, Recall, AUC)
- Automated training with callbacks (Early Stopping, Learning Rate Reduction)
- Visualization of training history

## 📊 Dataset

### Source
The dataset is sourced from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney.

### Dataset Structure
The dataset contains **5,863 chest X-ray images** (JPEG) organized into 3 folders:
- **Train**: 5,216 images
- **Test**: 624 images  
- **Validation**: 16 images

Each folder contains two subdirectories:
- `NORMAL`: X-rays from healthy patients
- `PNEUMONIA`: X-rays from patients with pneumonia

### Dataset Statistics
- Training set: 74% pneumonia, 26% normal
- Test set: 62% pneumonia, 38% normal
- Validation set: 50% pneumonia, 50% normal

## 📁 Project Structure

- **DownloadDataSet.py** - Script to download the Kaggle dataset
- **TrainingModel.py** - Script to train the CNN model  
- **README.md** - Project documentation
- **Generated files after training:**
  - best_pneumonia_model.h5 - Best model checkpoint
  - pneumonia_detection_model.h5 - Final trained model
  - training_history.png - Training/validation curves

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.x
- Kaggle API credentials

### Python Libraries
tensorflow>=2.10.0
kagglehub
numpy
matplotlib

## 🚀 Installation

### Step 1: Clone the Repository
git clone https://github.com/LilKostaa/Pneumonia_ML_Model.git
cd Pneumonia_ML_Model

### Step 2: Install Dependencies

### Step 3: Set Up Kaggle API

To download the dataset, you need to configure Kaggle API credentials:

1. **Create a Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)

2. **Generate API Token**:
   - Go to your [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to the **API** section
   - Click **"Create New API Token"**
   - This will download a `kaggle.json` file

3. **Place the API Token**:
   - **Linux/Mac**: Move `kaggle.json` to `~/.kaggle/`
   - **Windows**: Move to `C:\Users\<YourUsername>\.kaggle\`
   

## 💻 Usage

### 1. Download Dataset

Run the `DownloadDataSet.py` script to download the chest X-ray dataset from Kaggle:

**What it does:**
- Authenticates with Kaggle API using your credentials
- Downloads the latest version of the chest X-ray pneumonia dataset
- Prints the local path where the dataset is stored
- The dataset will be cached locally for future use

**Output:**
Path to dataset files: /path/to/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/...

### 2. Train the Model
Run the `TrainingModel.py` script to train the pneumonia detection model:

**What it does:**
- Downloads the dataset (if not already downloaded)
- Loads and preprocesses the images with data augmentation
- Builds a CNN model using VGG16 Transfer Learning
- Trains the model for 25 epochs with callbacks
- Saves the best model as `best_pneumonia_model.h5`
- Saves the final model as `pneumonia_detection_model.h5`
- Evaluates performance on the test set
- Generates training history visualization

**Training Process:**
1. **Data Loading**: Images are loaded and split into batches
2. **Data Augmentation**: Training images undergo rotation, shifting, zoom, and flipping
3. **Model Training**: VGG16-based model trains with early stopping
4. **Model Evaluation**: Test set metrics are computed
5. **Visualization**: Training/validation curves are saved

**Expected Output:**
Classes found: {'NORMAL': 0, 'PNEUMONIA': 1}
Training images: 5216
Validation images: 16
Test images: 624

Building model...
Starting training...
Epoch 1/25
...
Training completed successfully!

Test Accuracy: 0.XXXX
Test Precision: 0.XXXX
Test Recall: 0.XXXX
Test AUC: 0.XXXX


## 🏗️ Model Architecture

### Base Architecture: VGG16
- Pre-trained on ImageNet
- Last 4 layers unfrozen for fine-tuning
- Input size: 224×224×3

### Custom Layers
VGG16 Base Model (frozen layers)
↓
Global Average Pooling 2D
↓
Dense (512 units, ReLU) + Dropout (0.5)
↓
Dense (256 units, ReLU) + Dropout (0.3)
↓
Dense (128 units, ReLU) + Dropout (0.2)
↓
Dense (1 unit, Sigmoid) → Binary Classification


### Training Configuration
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Batch Size**: 32
- **Epochs**: 25 (with early stopping)

### Data Augmentation
- Rotation: ±20°
- Width/Height Shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal Flip: Yes

### Callbacks
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if validation loss doesn't improve for 5 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus

## 📈 Results

After training, the model generates:

1. **Model Files**:
   - `best_pneumonia_model.h5`: Best performing model during training
   - `pneumonia_detection_model.h5`: Final trained model

2. **Performance Metrics**:
   - Test Accuracy
   - Test Precision
   - Test Recall
   - Test AUC Score

3. **Visualization**:
   - `training_history.png`: Plots showing accuracy and loss curves over epochs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. Please refer to the dataset's license on Kaggle for usage terms.

## 🙏 Acknowledgments

- Dataset provided by [Paul Mooney](https://www.kaggle.com/paultimothymooney) on Kaggle
- Based on research from Kermany et al. on pediatric chest X-ray analysis
- VGG16 architecture by Visual Geometry Group, University of Oxford

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. It should not be used as a substitute for professional medical diagnosis.
