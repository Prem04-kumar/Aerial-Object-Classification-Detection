# 🛸 Aerial Object Classification & Detection
## Bird vs Drone — Custom CNN + Transfer Learning (EfficientNetB0)

## 📌 Project Overview
This project builds a deep learning model to classify aerial objects as either Bird 🐦 or Drone 🚁 using:

## Custom CNN — Built from scratch
## EfficientNetB0 — Transfer Learning from ImageNet
## Streamlit Web App — Real-time classification


## 🎯 Results
* ModelTest AccuracyCustom CNN~67%EfficientNetB0 (Transfer Learning)~98.6% ✅
* Best Model: EfficientNetB0 with 98.6% accuracy

## 📁 Project Structure
├── aerial_detection_final.ipynb  # Complete training notebook
│                                 # (Data loading, preprocessing,
│                                 #  Custom CNN, EfficientNetB0,
│                                 #  Evaluation & Model Comparison)
├── streamlit_app.py              # Streamlit web application
│                                 # (Upload image & classify Bird/Drone)
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── models/
│   └── best_model.keras          # Trained EfficientNetB0 model (see below)
└── classification_dataset/       # Image dataset (see below)
    ├── train/
    │   ├── bird/   (1414 images)
    │   └── drone/  (1248 images)
    ├── valid/
    │   ├── bird/   (217 images)
    │   └── drone/  (225 images)
    └── test/
        ├── bird/   (121 images)
        └── drone/  (94 images)

## 📦 Dataset
Download the classification dataset from Google Drive:
🔗 https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh
After downloading, place it in the project folder as:
classification_dataset/
├── train/
├── valid/
└── test/

## 🧠 Pre-trained Model
Download the trained model from Google Drive:
🔗 (https://drive.google.com/file/d/1N5QdvT9tCWNflj7gQJJ14xVr9Kjw9A4V/view?usp=sharing)
After downloading, place it in:
models/best_model.keras

# 🚀 How to Run
## Step 1 — Clone the repository
bashgit clone https://github.com/your-username/aerial-detection.git
cd aerial-detection
## Step 2 — Install dependencies
bashpip install -r requirements.txt
## Step 3 — Run the Streamlit app
bashstreamlit run streamlit_app.py
## Step 4 — Open in browser
http://localhost:8501


# 📊 Model Architecture
## Custom CNN

* 3 Convolutional blocks with BatchNormalization & Dropout
* GlobalAveragePooling2D
* Dense layers with Sigmoid output

## EfficientNetB0 (Transfer Learning)

* Pre-trained on ImageNet (frozen base)
* Custom classification head
* Fine-tuned for Bird vs Drone classification


## 📈 Training Details
ParameterValueImage Size224 × 224Batch Size32OptimizerAdamLossBinary CrossentropyEpochs30 (EarlyStopping)

## 🌐 Streamlit App Features

***** Upload any Bird or Drone image (JPG/PNG)
******* Real-time classification with confidence score
********Probability distribution chart
****Supports EfficientNet preprocessing

## 👨‍💻 Author
A Prem Kumar
