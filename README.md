# 👗 ClosetCraft: AI Fashion Recommendation & Outfit Generator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **An AI-powered fashion recommendation system using Transfer Learning with ResNet50 and Cosine Similarity**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Adding Your Own Dataset](#-adding-your-own-dataset)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

**ClosetCraft** is an intelligent fashion recommendation system that uses **Transfer Learning** and **Computer Vision** to suggest visually similar clothing items. Instead of training a model from scratch, it leverages a pretrained **ResNet50** network (trained on ImageNet) as a feature extractor to generate deep feature vectors from clothing images.

The system then uses **Cosine Similarity** to find and recommend the most visually similar items from a custom fashion dataset.

---

## ✨ Features

- ✅ **Image Upload** - Upload any clothing image via web interface
- ✅ **AI Feature Extraction** - Uses ResNet50 to extract deep visual features
- ✅ **Smart Recommendations** - Finds visually similar items using Cosine Similarity
- ✅ **Outfit Generation** - Suggests complete outfit combinations
- ✅ **Color Compatibility** - Basic color analysis for better recommendations
- ✅ **Web Interface** - Clean Flask-based UI with HTML/CSS
- ✅ **Lightweight** - No heavy model training required
- ✅ **Extensible** - Easy to add new clothing items

---

## 🧠 How It Works
┌─────────────────┐
│ User Uploads │
│ Clothing Image │
└────────┬────────┘
▼
┌─────────────────┐
│ ResNet50 │
│ Feature │
│ Extraction │
└────────┬────────┘
▼
┌─────────────────┐
│ Feature │
│ Vector │
│ (2048-d) │
└────────┬────────┘
▼
┌─────────────────┐
│ Cosine │
│ Similarity │
│ Comparison │
└────────┬────────┘
▼
┌─────────────────┐
│ Top 5 Similar │
│ Items │
│ Recommended │
└─────────────────┘

### Technical Workflow:

1. **Feature Extraction**: Each image → ResNet50 (pretrained) → 2048-dimensional feature vector
2. **Dataset Building**: All dataset images processed → feature vectors stored in `dataset_features.pkl`
3. **Similarity Search**: User image → feature vector → Cosine similarity with dataset → Top matches
4. **Recommendation**: Returns visually similar items with confidence scores

---

## 🛠️ Tech Stack

### Machine Learning
- **PyTorch** - Deep learning framework
- **Torchvision** - Pre-trained models & image transformations
- **ResNet50** - Feature extractor (Transfer Learning)
- **Scikit-learn** - Cosine similarity computation
- **NumPy** - Numerical operations

### Backend
- **Flask** - Web framework
- **Werkzeug** - File handling & utilities

### Frontend
- **HTML5**
- **CSS3**
- **JavaScript** (basic)

### Utilities
- **OpenCV** - Image processing
- **Pillow** - Image handling
- **Pickle** - Feature caching

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/closetcraft.git
cd closetcraft
