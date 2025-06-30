# cerebrolens1
# 🧠 CerebroLens - Brain Tumor Detection & Classification System

CerebroLens is a full-stack AI-powered application that performs **brain tumor segmentation and classification** using deep learning. The system integrates **U-Net** for precise tumor region detection and **EfficientNet** for accurate classification into **glioma**, **meningioma**, **pituitary**, or **no tumor**. A standout feature of the app is the **preliminary biopsy suggestion module**, designed to assist healthcare professionals with timely decision-making.

## 🚀 Key Features

- 🔍 **Tumor Detection** using U-Net Segmentation
- 🧪 **Tumor Classification** using EfficientNet
- 📝 **Biopsy Suggestion Module** *(experimental but promising for future medical integrations)*
- 📊 **Model Accuracy Page** showcasing evaluation metrics
- 👤 **Secure Login/Signup System**
- 📥 **Downloadable Report** for medical consultation and sharing
- 🌐 **Streamlit-based Intuitive Interface** – no backend server/API calls required

  
- **Frontend & UI**: Streamlit
- **Models**:
  - U-Net for segmentation (BraTS 2020 dataset)
  - EfficientNet for classification (Kaggle brain tumor dataset)
- **Authentication**: Python + Streamlit session management
- **Reporting**: Auto-generated diagnosis reports in PDF/Word format
- **Languages**: Python 3.9+

## 🧪 Installation & Setup

### 🔧 Prerequisites

- Python ≥ 3.8
- pip
- Virtual environment (recommended)

### ⚙️ Installation

```bash
# Clone the repository

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run cerbrolens.py
