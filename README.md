# 🛸 Aerial Object Classification (Bird 🐦 vs Drone 🚁)

## 📌 Project Overview
This project focuses on classifying aerial objects as either a **Bird** or a **Drone** using deep learning techniques.

It combines:
- 🧠 Custom CNN (built from scratch)
- ⚡ Transfer Learning with EfficientNetB0
- 🌐 Streamlit Web App for real-time predictions

---

## 🎯 Results

| Model           | Type                | Test Accuracy |
|-----------------|--------------------|--------------|
| Custom CNN      | Built from scratch | ~67%         |
| EfficientNetB0  | Transfer Learning  | **~98.6% ✅** |

🏆 **Best Model:** EfficientNetB0 (98.6% accuracy)

---

## 📁 Project Structure
C:/VSCODE/Aerial_project/
│
├── streamlit_app.py # Streamlit deployment script
├── aerial_object.ipynb # Model training & EDA
├── best_model.keras # Trained EfficientNetB0 model
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
└── classification_dataset/
├── train/
│ ├── bird/
│ └── drone/
├── valid/
│ ├── bird/
│ └── drone/
└── test/
├── bird/
└── drone/


---

## 📦 Dataset

📥 Download Dataset:  
👉 [(https://drive.google.com/drive/folders/1_59wS79EcA3x6ojVCeNIIGXp9rQRclNO?usp=drive_link)]

After downloading, place it like this:


---

## 🧠 Pre-trained Model

📥 Download Model:  
👉 [(https://drive.google.com/file/d/1N5QdvT9tCWNflj7gQJJ14xVr9Kjw9A4V/view?usp=drive_link)]

Place the model file at:
C:\VSCODE\Aerial_project\best_model.keras


---

## 🚀 How to Run

### 1️⃣ Setup Environment
cd C:\VSCODE\Aerial_project


### 2️⃣ Install Dependencies
pip install -r requirements.txt


### 3️⃣ Run Streamlit App
streamlit run streamlit_app.py


---

## 📊 Model Architecture

### 🔹 Custom CNN
- 3 Convolutional Layers
- Batch Normalization
- Dropout Regularization
- GlobalAveragePooling2D
- Dense + Sigmoid Output

### 🔹 EfficientNetB0 (Transfer Learning)
- Pre-trained on ImageNet
- Frozen base layers
- Custom classification head
- Fine-tuned for Bird vs Drone

---

## 📈 Training Details

| Parameter   | Value                |
|------------|---------------------|
| Image Size | 224 × 224           |
| Batch Size | 32                  |
| Optimizer  | Adam                |
| Loss       | Binary Crossentropy |
| Epochs     | 30 (EarlyStopping)  |

---

## 🌐 Streamlit App Features

- 📤 Upload Image (JPG, JPEG, PNG)
- ⚡ Real-time Prediction
- 📊 Confidence Score Visualization
- 🔄 Preprocessing Options (EfficientNet / MobileNet)

---

## 🖼️ Demo

*(Add screenshots of your Streamlit app here)*

---

## 💡 Use Cases

- 🛡️ Surveillance Systems
- ✈️ Airspace Monitoring
- 🌿 Wildlife Observation
- 🚁 Drone Detection & Security

---

## 👨‍💻 Author

**A Prem Kumar**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
