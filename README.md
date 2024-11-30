# 🤖 ModelWise - ML Model Selection Advisor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/PyQt-6.4.0-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A desktop application that helps you select the most appropriate machine learning models by analyzing your dataset characteristics and providing AI-powered recommendations.

## ✨ Features

- 📊 Interactive Dataset Analysis
  - Sample dataset exploration (Iris, Breast Cancer, Synthetic datasets)
  - Custom dataset upload (CSV)
  - Real-time data preview
  - Dataset statistics and characteristics

- 🎯 Smart Model Recommendations
  - Dataset size-based suggestions
  - Model complexity analysis
  - Detailed pros and cons
  - Visual recommendation cards

- 🤖 AI Assistant
  - Interactive chat interface powered by Google's Gemini Pro
  - Context-aware model explanations
  - Dataset-specific guidance
  - ML concept clarification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MODELWISE.git
cd MODELWISE
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages

```
PyQt6==6.4.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.2
google-generativeai>=0.3.0
toml>=0.10.2
```

## 💡 Usage

1. Set up your Google AI API key in `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your-api-key-here"
```

2. Launch the desktop app:
```bash
python modelwise_desktop.py
```

3. Use the interface to:
   - Choose from sample datasets or upload your own CSV
   - View dataset characteristics and preview
   - Get model recommendations
   - Chat with the AI assistant for guidance

## 🖥️ Interface

The application features a clean, modern interface with:
- Left panel: Dataset selection and information
- Center panel: Data preview and model recommendations
- Right panel: AI chat assistant

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/MODELWISE/issues).

## 📝 License

This project is [MIT](LICENSE) licensed.

---

<p align="center">Made with ❤️ by Kyle</p>