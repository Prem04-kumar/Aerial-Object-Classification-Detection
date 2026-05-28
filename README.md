# 🛸 Aerial Object Classification & Detection — Bird 🐦 vs Drone 🚁
 
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNetB0-Transfer%20Learning-4CAF50?style=for-the-badge)
 
---
 
## 📌 Project Overview
 
This project focuses on classifying aerial objects as either a **Bird** or a **Drone** using deep learning techniques.
 
It combines:
 
- 🧠 **Custom CNN** — built from scratch for baseline performance
- ⚡ **Transfer Learning** with EfficientNetB0 — for high-accuracy production-ready results
- 🌐 **Streamlit Web App** — for real-time image upload and prediction
> This system has direct applications in **surveillance, airspace monitoring, wildlife observation, and drone security**.
 
---
 
## 🎯 Business Problem
 
With the rapid increase in drone usage, distinguishing drones from birds in aerial footage has become critical for:
 
| Challenge | Impact |
|-----------|--------|
| ✈️ Airspace safety | Prevent collisions and unauthorized drone flights |
| 🛡️ Security surveillance | Detect intrusions in restricted zones |
| 🌿 Wildlife monitoring | Avoid disrupting birds during drone operations |
| 🚁 Counter-drone systems | Automate real-time aerial threat detection |
 
Manual identification is slow and error-prone — this project automates it using **Deep Learning**.
 
---
 
## 🏆 Results
 
| Model | Type | Test Accuracy |
|-------|------|---------------|
| Custom CNN | Built from scratch | ~67% |
| **EfficientNetB0** | **Transfer Learning** | **~98.6% ✅** |
 
> 🥇 **Best Model:** EfficientNetB0 with **98.6% accuracy**
 
---
 
## 🛠️ Technologies Used
 
| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras |
| Transfer Learning | EfficientNetB0 (ImageNet) |
| Data Augmentation | Keras ImageDataGenerator |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Environment | Jupyter Notebook |
 
---
 
## 📦 Dataset
 
📥 **Download Dataset:**
👉 [Google Drive — Classification Dataset](https://drive.google.com/drive/folders/1_59wS79EcA3x6ojVCeNIIGXp9rQRclNO?usp=drive_link)
 
After downloading, organize the dataset as follows:
 
```
classification_dataset/
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
 
---
 
## 🧠 Pre-trained Model
 
📥 **Download Trained Model:**
👉 [Google Drive — best_model.keras](https://drive.google.com/file/d/1N5QdvT9tCWNflj7gQJJ14xVr9Kjw9A4V/view?usp=drive_link)
 
Place the downloaded model at:
 
```
Aerial-Object-Classification-Detection/
└── best_model.keras
```
 
---
 
## 📁 Project Structure
 
```
Aerial-Object-Classification-Detection/
│
├── aerial_object.ipynb          # Model training, EDA & evaluation notebook
├── streamlit_app.py             # Streamlit deployment script
├── best_model.keras             # Trained EfficientNetB0 model (download separately)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── classification_dataset/      # Dataset folder (download separately)
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
 
---
 
## 🔄 Project Workflow
 
```
Dataset (Bird / Drone Images)
        ↓
Data Preprocessing & Augmentation
        ↓
  ┌─────────────────────────────┐
  │   Model 1: Custom CNN       │  → ~67% Accuracy
  │   Model 2: EfficientNetB0   │  → ~98.6% Accuracy ✅
  └─────────────────────────────┘
        ↓
Model Evaluation (Accuracy, Loss, Confusion Matrix)
        ↓
Best Model Saved (best_model.keras)
        ↓
Streamlit Web App Deployment
        ↓
Real-time Bird vs Drone Prediction
```
 
---
 
## 📊 Model Architecture
 
### 🔹 Model 1 — Custom CNN (Baseline)
 
Built entirely from scratch:
 
| Layer | Details |
|-------|---------|
| Conv2D (×3) | Feature extraction layers |
| BatchNormalization | Stabilize training |
| MaxPooling2D | Spatial downsampling |
| Dropout | Regularization to prevent overfitting |
| GlobalAveragePooling2D | Flatten spatial features |
| Dense + Sigmoid | Binary classification output |
 
> Achieved **~67% accuracy** — a solid baseline but limited by training data size.
 
---
 
### 🔹 Model 2 — EfficientNetB0 (Transfer Learning)
 
Leverages a model pre-trained on **ImageNet (1.2M images)**:
 
| Component | Details |
|-----------|---------|
| Base Model | EfficientNetB0 (frozen ImageNet weights) |
| Fine-Tuning | Top layers unfrozen and retrained |
| Custom Head | GlobalAveragePooling2D → Dense → Dropout → Sigmoid |
| Output | Binary (Bird / Drone) |
 
> Achieved **~98.6% accuracy** — production-ready performance.
 
---
 
## 📈 Training Details
 
| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 30 (with EarlyStopping) |
| Augmentation | Rotation, Flip, Zoom, Shear, Brightness |
 
---
 
## 🌐 Streamlit App Features
 
| Feature | Description |
|---------|-------------|
| 📤 Image Upload | Supports JPG, JPEG, PNG formats |
| ⚡ Real-time Prediction | Instant Bird vs Drone classification |
| 📊 Confidence Score | Probability-based confidence visualization |
| 🔄 Preprocessing Options | EfficientNet / MobileNet compatible |
| 🖼️ Preview | Displays uploaded image alongside results |
 
---
 
## 🚀 How to Run
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Prem04-kumar/Aerial-Object-Classification-Detection.git
cd Aerial-Object-Classification-Detection
```
 
### 2. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Download Dataset & Model
 
- Download the [dataset](https://drive.google.com/drive/folders/1_59wS79EcA3x6ojVCeNIIGXp9rQRclNO?usp=drive_link) and place in `classification_dataset/`
- Download [best_model.keras](https://drive.google.com/file/d/1N5QdvT9tCWNflj7gQJJ14xVr9Kjw9A4V/view?usp=drive_link) and place in the root folder
### 4. Train the Model (Optional)
 
```bash
jupyter notebook aerial_object.ipynb
```
 
### 5. Run the Streamlit App
 
```bash
streamlit run streamlit_app.py
```
 
Open your browser at `http://localhost:8501` and upload an image to classify!
 
---

 
---
 
## 💡 Use Cases
 
| Domain | Application |
|--------|-------------|
| 🛡️ Surveillance Systems | Detect unauthorized drones in restricted areas |
| ✈️ Airspace Monitoring | Identify aerial objects near airports |
| 🌿 Wildlife Observation | Distinguish birds from drones in nature reserves |
| 🚁 Counter-Drone Security | Automate drone threat classification in real-time |
| 🏟️ Event Security | Monitor large public venues for drone intrusions |
 
---
 
## 📊 Model Comparison
 
```
Accuracy
100% |                          ██████████
 98% |                          ██████████  ← EfficientNetB0 (98.6%)
 80% |
 67% |  ██████████
 60% |  ██████████  ← Custom CNN (67%)
 40% |
     |________________________
          Custom CNN     EfficientNetB0
```
 
---
 
## 🌐 Future Enhancements
 
- [ ] 🎥 Real-time **video stream** classification
- [ ] 📦 Add **multi-class detection** (Birds, Drones, Helicopters, Planes)
- [ ] 🔍 Integrate **YOLOv8** for object detection with bounding boxes
- [ ] ☁️ Deploy on **AWS / GCP / Hugging Face Spaces**
- [ ] 📱 Build a **mobile app** for field surveillance use
- [ ] 🔔 Add **alert system** for drone detection events
---
 
## 📚 Key Learnings
 
- ✅ Building CNN architectures from scratch
- ✅ Transfer Learning with EfficientNetB0
- ✅ Data Augmentation for small datasets
- ✅ Binary image classification pipeline
- ✅ Model fine-tuning and layer unfreezing
- ✅ Streamlit app development and deployment
- ✅ Confidence score visualization
- ✅ Model saving and loading (`.keras` format)
---
 
## 👨‍💻 Author
 
**Prem Kumar A**
 
[![GitHub](https://img.shields.io/badge/GitHub-Prem04--kumar-181717?style=for-the-badge&logo=github)](https://github.com/Prem04-kumar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/)
 
> 📁 Project Type: **Deep Learning | Computer Vision | End-to-End Deployment**
 
---
 
## 📌 Conclusion
 
This project demonstrates the power of **Transfer Learning** for real-world computer vision tasks. By leveraging EfficientNetB0 pre-trained on ImageNet, the model achieves **98.6% accuracy** in distinguishing birds from drones — far surpassing the baseline CNN's 67%.
 
The complete pipeline — from data preprocessing to Streamlit deployment — makes this a fully production-ready aerial object classification system.
 
---
 
> ⭐ **If you found this project helpful, please give it a star on GitHub!**
