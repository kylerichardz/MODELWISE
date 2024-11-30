# 🤖 ML Model Selection Advisor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent tool that helps you select the most appropriate machine learning models by analyzing your dataset characteristics and providing AI-powered recommendations.

## ✨ Features

- 📊 Interactive Dataset Analysis
  - Sample dataset exploration
  - Custom dataset upload (CSV)
  - Basic statistics and data preview
  - Dataset characteristics visualization

- 🎯 Smart Model Recommendations
  - Context-aware model suggestions
  - Detailed pros and cons for each model
  - Visual recommendation cards
  - Tailored for different dataset types

- 🤖 AI Assistant
  - Interactive chat interface powered by Google's Gemini Pro
  - Real-time model explanations
  - Dataset guidance
  - ML concept clarification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-model-advisor.git
cd ml-model-advisor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages

```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
google-generativeai>=0.1.0
plotly>=5.0.0
```

## 💡 Usage

1. Set up your Google AI API key:
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

2. Launch the app:
```bash
streamlit run app.py
```

3. Use the interface to:
   - Choose from sample datasets or upload your own CSV
   - Review dataset characteristics and statistics
   - Get model recommendations
   - Chat with the AI assistant for guidance

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/ml-model-advisor/issues).

## 📝 License

This project is [MIT](LICENSE) licensed.

---

<p align="center">Made with ❤️ by Kyle</p>