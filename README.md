AI Powered histopathologic Cancer Detector WebApp:

An advanced, end-to-end deep learning application for detecting metastatic cancer in histopathologic image patches. This project features a custom-built Hybrid Attention-CNN Transformer (HACT) model, served via a Flask backend, with a dynamic, custom-built frontend.

🚀 Key Features
Advanced AI Model: Utilizes a novel Hybrid Attention-CNN Transformer (HACT) architecture, combining the feature extraction power of EfficientNetV2 with the contextual understanding of a Transformer.

Interactive Web Interface: A dynamic, user-friendly frontend built with HTML, Tailwind CSS, and vanilla JavaScript, allowing for easy image upload and analysis.

Full End-to-End Pipeline: The project covers the complete machine learning lifecycle, from data exploration and model training in Kaggle notebooks to a fully deployed web application.

Containerized Deployment: The entire application is containerized using Docker, ensuring consistent and reliable deployment on platforms like Hugging Face Spaces.

🛠️ Project Structure
.
├── 📄 .gitignore
├── 🐳 Dockerfile
├── 🖼️ app-screenshot.png
├── 🐍 flask_app.py
├── 🌐 index.html
├── 🔬 hact_model_epoch_10.pth
├── 📋 README.md
├── 📦 requirements.txt
│
└── 📁 kaggle-development/
    ├── 📓 Main-Cancer-Detection-Trainer.ipynb
    ├── 🐍 hact_model.py
    ├── 🐍 pcam_data_utils.py
    └── 🐍 training_utils.py

💻 Running Locally
To run this application on your local machine, follow these steps.

Step 1: Clone the Repository

git clone [https://github.com/asanepranav/AI-Cancer-Detector-WebApp.git](https://github.com/asanepranav/AI-Cancer-Detector-WebApp.git)
cd AI-Cancer-Detector-WebApp

Step 2: Set Up the Python Environment

It is highly recommended to use a virtual environment.

# Create and activate a Conda environment
conda create -n cancer-app python=3.9 -y
conda activate cancer-app

# Install the necessary dependencies
pip install -r requirements.txt

Step 3: Run the Backend Server

The backend is a Flask application that serves the AI model.

# Run the Flask application
flask --app flask_app run

The server will start and be available at http://127.0.0.1:5000.

Step 4: Launch the Frontend

Open the index.html file in your web browser to use the application.

🚀 Deployment on Hugging Face
This application is deployed as a Docker Space on Hugging Face.

Step 1: Create a New Space

Log in to Hugging Face and create a New Space.

Choose a Space name and License (e.g., MIT).

Select the Space SDK as Docker.

Link it directly to your GitHub repository by pasting the URL in the "Clone from a GitHub repo" field.

Choose the "CPU basic" hardware (free tier).

Click "Create Space".

Hugging Face will automatically clone the repository, find the Dockerfile, and deploy the application.
