from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QComboBox, QPushButton, QLabel, 
                           QTableWidget, QTableWidgetItem, QFileDialog,
                           QTextEdit, QLineEdit, QScrollArea)
from PyQt6.QtCore import Qt
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
import google.generativeai as genai
import toml
from pathlib import Path

class ModelWiseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ModelWise - ML Model Selection Advisor")
        self.setMinimumSize(1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for dataset selection and info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Dataset selection
        dataset_label = QLabel("Choose Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Iris (Small Classification)",
            "Breast Cancer (Medium Classification)",
            "Synthetic (Large Classification)",
            "Synthetic Regression (Large)"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Upload button
        self.upload_btn = QPushButton("Upload CSV")
        self.upload_btn.clicked.connect(self.upload_csv)
        
        # Dataset info
        self.info_label = QLabel("Dataset Information:")
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        # Add widgets to left panel
        left_layout.addWidget(dataset_label)
        left_layout.addWidget(self.dataset_combo)
        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(self.info_label)
        left_layout.addWidget(self.info_table)
        
        # Right panel for data preview and recommendations
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Data preview
        preview_label = QLabel("Data Preview:")
        self.data_table = QTableWidget()
        
        # Model recommendations
        recommendations_label = QLabel("Model Recommendations:")
        self.recommendations_text = QLabel()
        self.recommendations_text.setWordWrap(True)
        
        # Add widgets to right panel
        right_layout.addWidget(preview_label)
        right_layout.addWidget(self.data_table)
        right_layout.addWidget(recommendations_label)
        right_layout.addWidget(self.recommendations_text)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        
        # Configure Google AI
        try:
            # Load secrets from .streamlit/secrets.toml
            secrets_path = Path('.streamlit/secrets.toml')
            if secrets_path.exists():
                secrets = toml.load(secrets_path)
                api_key = secrets.get('GOOGLE_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-pro')
                    self.chat_enabled = True
                else:
                    print("API key not found in secrets.toml")
                    self.chat_enabled = False
            else:
                print("secrets.toml file not found")
                self.chat_enabled = False
        except Exception as e:
            print(f"Error configuring chat: {e}")
            self.chat_enabled = False

        # Add chat panel
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        
        # Chat history
        chat_scroll = QScrollArea()
        chat_scroll.setWidgetResizable(True)
        chat_content = QWidget()
        self.chat_layout = QVBoxLayout(chat_content)
        chat_scroll.setWidget(chat_content)
        
        # Chat input
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about your data or models...")
        self.chat_input.returnPressed.connect(self.send_message)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(send_button)
        
        # Add to chat panel
        chat_layout.addWidget(QLabel("AI Assistant"))
        chat_layout.addWidget(chat_scroll)
        chat_layout.addLayout(chat_input_layout)
        
        # Add chat panel to main layout
        layout.addWidget(chat_panel, 1)
        
        # Load initial dataset
        self.load_dataset()

    def load_dataset(self):
        option = self.dataset_combo.currentText()
        
        if option == "Iris (Small Classification)":
            data = load_iris()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            dataset_type = "Classification"
            
        elif option == "Breast Cancer (Medium Classification)":
            data = load_breast_cancer()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df['target'] = data.target
            dataset_type = "Classification"
            
        elif option == "Synthetic (Large Classification)":
            X, y = make_classification(n_samples=10000, n_features=20, 
                                    n_informative=15, n_redundant=5, random_state=42)
            self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            self.df['target'] = y
            dataset_type = "Classification"
            
        else:  # Synthetic Regression
            X = np.random.randn(10000, 10)
            y = np.sum(X[:, :3], axis=1) + np.random.randn(10000) * 0.1
            self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            self.df['target'] = y
            dataset_type = "Regression"
        
        self.update_info(dataset_type)
        self.update_preview()
        self.update_recommendations()

    def upload_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.df = pd.read_csv(file_name)
            self.update_info("Unknown")
            self.update_preview()
            self.update_recommendations()

    def update_info(self, dataset_type):
        info = [
            ("Samples", len(self.df)),
            ("Features", len(self.df.columns) - 1),
            ("Type", dataset_type),
            ("Missing Values", self.df.isnull().sum().sum())
        ]
        
        self.info_table.setRowCount(len(info))
        for i, (metric, value) in enumerate(info):
            self.info_table.setItem(i, 0, QTableWidgetItem(str(metric)))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def update_preview(self):
        # Update data preview table
        self.data_table.setRowCount(5)  # Show first 5 rows
        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(5):
            for j in range(len(self.df.columns)):
                self.data_table.setItem(i, j, 
                    QTableWidgetItem(str(self.df.iloc[i, j])))

    def update_recommendations(self):
        n_samples = len(self.df)
        
        if n_samples < 1000:
            recommendations = """
            For small datasets (<1000 samples):
            • Linear/Logistic Regression (Simple & Interpretable)
            • Decision Trees (Easy to Understand)
            """
        elif n_samples < 10000:
            recommendations = """
            For medium datasets:
            • Random Forest (Robust & Accurate)
            • Support Vector Machines (Powerful for Complex Patterns)
            """
        else:
            recommendations = """
            For large datasets:
            • Gradient Boosting (XGBoost, LightGBM)
            • Neural Networks (Deep Learning)
            """
            
        self.recommendations_text.setText(recommendations)

    def send_message(self):
        if not self.chat_input.text().strip():
            return
            
        # Add user message to chat
        user_message = self.chat_input.text()
        self.add_message("User", user_message)
        
        if self.chat_enabled:
            try:
                # Get context about current data
                context = f"""
                Current Dataset: {self.dataset_combo.currentText()}
                Number of samples: {len(self.df)}
                Number of features: {len(self.df.columns) - 1}
                """
                
                # Generate response
                prompt = f"""
                Context: {context}
                User Question: {user_message}
                Please provide advice based on this context and your knowledge of machine learning.
                """
                
                response = self.model.generate_content(prompt)
                self.add_message("Assistant", response.text)
            except Exception as e:
                self.add_message("Assistant", f"Error generating response: {str(e)}")
        else:
            self.add_message("Assistant", "Chat is currently disabled. Please check your API key configuration.")
        
        self.chat_input.clear()

    def add_message(self, sender, text):
        message_widget = QLabel(f"{sender}: {text}")
        message_widget.setWordWrap(True)
        message_widget.setStyleSheet(
            """
            color: #000000;
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
            font-size: 12px;
            """ if sender == "User" else """
            color: #000000;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
            font-size: 12px;
            """
        )
        self.chat_layout.addWidget(message_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelWiseApp()
    window.show()
    sys.exit(app.exec()) 