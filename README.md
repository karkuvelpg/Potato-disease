# 🥔 Potato Disease Detection & Classification

An end-to-end **Potato Leaf Disease Classification** project that uses **Deep Learning** to detect whether potato plant leaves are **healthy** or affected by diseases.  
This project includes a trained model, API endpoints, and demo scripts for testing and classification.

---

## 📌 Project Overview

This project performs:

- 📸 Image-based classification of potato leaf disease
- 🤖 Deep learning model trained using **Keras / TensorFlow**
- 🧪 Demo and testing using sample images
- 🚀 REST API to serve predictions using **FastAPI**
- 🐍 GUI/CLI script (`app.py`) to classify disease locally
- 📦 Requirements management with `requirements.txt`

The dataset used for training contains images of healthy and diseased potato leaves (such as **Early Blight** and **Late Blight**) commonly found in *PlantVillage-type datasets* (e.g., from Kaggle). :contentReference[oaicite:1]{index=1}

---

## 🧠 Key Features

- 🥔 **Healthy vs Diseased Classification**
- 🔍 Uses convolutional neural network (CNN) methods
- 🧪 Test images included in `test_img/`
- 🔌 **FastAPI backend** (`fast_api.py`) to serve predictions
- 📌 Demo script (`app.py`) to classify one image at a time
- 🧾 Requirements file for easy setup

---

## 📂 Project Structure

```

Potato-disease/
│
├── PlantVillage/                   # Training/test data folder
├── Potato_Disease_Classification.ipynb
├── app.py                          # Model testing script
├── fast_api.py                     # API endpoints
├── potato_models_h5_file/          # Trained model files
├── Potato_models_keras_file/       # Model weights & metadata
├── test_img/                       # Sample images for testing
├── requirements.txt
└── README.md

````

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** – Deep Learning
- **FastAPI** – REST API backend
- **Jupyter Notebook** – Training & experimentation
- **NumPy, OpenCV, Pillow** – Image handling

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/karkuvelpg/Potato-disease.git
cd Potato-disease
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Notebook (Training / Eval)

* Open **Potato_Disease_Classification.ipynb**
* Train or evaluate the model
* Visualize results and metrics

---

## 📌 Run Local Demo Script

To test a single image using the trained model:

```bash
python app.py --image test_img/your_image.jpg
```

*(Replace with your actual image path)*

---

## 🚀 Run FastAPI Server

Start the backend API:

```bash
uvicorn fast_api:app --reload
```

You can then send POST requests to the prediction endpoint to classify images.

---

## 🧠 Model Performance

This CNN-based classifier is trained to identify key diseases such as:

* **Early Blight**
* **Late Blight**
* **Healthy Leaves**

The dataset images include diverse samples of potato leaf patterns commonly used in agriculture image classification challenges. ([Kaggle][1])

---

## 🌟 Learning Outcomes

By working with this project, you will learn:

* Training deep learning models with **CNNs**
* Image preprocessing for machine learning
* Creating REST APIs with **FastAPI**
* Deploying model inference scripts
* Integrating machine learning with Python applications

---

## 🔮 Future Improvements

* 🧠 Model improvement & hyperparameter tuning
* 📊 Add evaluation metrics dashboard
* 📱 Web interface or mobile app
* 🛠️ Dockerize backend for easier deployment
* 📈 Add support for more plant diseases

---

## 👨‍💻 Author

**Karkuvel P**
M.Sc Mathematics | Data Science | Machine Learning

---

⭐ If you like this project, please **give it a star** on GitHub!

```
