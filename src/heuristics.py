from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelInfo:
    name: str
    description: str
    suitable_for: List[str]  # ['classification', 'regression']
    min_samples: int
    max_samples: int = float('inf')
    handles_missing_values: bool = False
    handles_categorical: bool = False
    handles_imbalanced: bool = False
    complexity: str = 'medium'  # 'low', 'medium', 'high'

# Define our model catalog
MODEL_CATALOG = {
    'linear_regression': ModelInfo(
        name='Linear Regression',
        description='Simple and interpretable model for regression tasks',
        suitable_for=['regression'],
        min_samples=30,
        handles_missing_values=False,
        handles_categorical=False,
        complexity='low'
    ),
    'logistic_regression': ModelInfo(
        name='Logistic Regression',
        description='Simple and interpretable model for classification tasks',
        suitable_for=['classification'],
        min_samples=30,
        handles_missing_values=False,
        handles_categorical=False,
        complexity='low'
    ),
    'random_forest': ModelInfo(
        name='Random Forest',
        description='Versatile ensemble model with good performance',
        suitable_for=['classification', 'regression'],
        min_samples=100,
        handles_missing_values=True,
        handles_categorical=True,
        handles_imbalanced=True,
        complexity='medium'
    ),
    'xgboost': ModelInfo(
        name='XGBoost',
        description='High-performance gradient boosting model',
        suitable_for=['classification', 'regression'],
        min_samples=1000,
        handles_missing_values=True,
        handles_categorical=False,
        handles_imbalanced=True,
        complexity='high'
    )
}

# Add more models as needed... 