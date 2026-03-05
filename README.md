# 🛸 Aerial Object Classification & Detection
## Bird vs Drone — Custom CNN + Transfer Learning (EfficientNetB0)

## 📌 Project Overview
This project builds a deep learning model to classify aerial objects as either Bird 🐦 or Drone 🚁 using:
- **Custom CNN — Built from scratch**
- **EfficientNetB0 — Transfer Learning from ImageNet**
- **Streamlit Web App — Real-time classification**

## 🎯 Results
## 📊 Model Performance Comparison

| Model | Type | Test Accuracy |
|------|------|---------------|
| Custom CNN | Built from scratch | ~67% |
| EfficientNetB0 | Transfer Learning | ~98.6% ✅ |

🏆 **Best Model:** EfficientNetB0 with **98.6% accuracy**

## 📁 Project Structure

```
aerial-detection/
│
├── aerial_detection_final.ipynb
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_model.keras
│
└── classification_dataset/
    ├── train/
    │   ├── bird/
    │   └── drone/
    ├── valid/
    │   ├── bird/
    │   └── drone/
    └── test/
        ├── bird/
        └── drone/
```

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
streamlit run streamlit_app.py



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

- **Upload any Bird or Drone image (JPG/PNG)**
- **Real-time classification with confidence score**
- **Probability distribution chart**
- **Supports EfficientNet preprocessing**

## 👨‍💻 Author
A Prem Kumar
