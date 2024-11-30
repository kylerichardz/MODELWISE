from typing import Dict, Any
import pandas as pd
import numpy as np

class DatasetAnalyzer:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    def analyze(self) -> Dict[str, Any]:
        """Analyze the dataset and return its characteristics"""
        return {
            'n_samples': len(self.df),
            'n_features': len(self.X.columns),
            'target_type': self._determine_target_type(),
            'has_missing_values': self.df.isnull().any().any(),
            'has_categorical': self._has_categorical_features(),
            'is_imbalanced': self._check_imbalanced(),
            'feature_types': self._get_feature_types()
        }

    def _determine_target_type(self) -> str:
        """Determine if this is a classification or regression problem"""
        if pd.api.types.is_numeric_dtype(self.y):
            unique_count = len(self.y.unique())
            if unique_count < 10:  # This is a heuristic
                return 'classification'
            return 'regression'
        return 'classification'

    def _has_categorical_features(self) -> bool:
        """Check if dataset has categorical features"""
        return any(pd.api.types.is_categorical_dtype(self.X[col]) or 
                  pd.api.types.is_object_dtype(self.X[col])
                  for col in self.X.columns)

    def _check_imbalanced(self) -> bool:
        """Check if classification dataset is imbalanced"""
        if self._determine_target_type() == 'classification':
            value_counts = self.y.value_counts()
            ratio = value_counts.min() / value_counts.max()
            return ratio < 0.2  # This is a heuristic
        return False

    def _get_feature_types(self) -> Dict[str, int]:
        """Count different types of features"""
        return {
            'numeric': len([col for col in self.X.columns 
                          if pd.api.types.is_numeric_dtype(self.X[col])]),
            'categorical': len([col for col in self.X.columns 
                              if pd.api.types.is_categorical_dtype(self.X[col]) or 
                              pd.api.types.is_object_dtype(self.X[col])])
        } 

def get_current_context():
    context = "Current App State:\n"
    
    if 'df' in locals():
        context += f"""
Dataset: {dataset_option}
Number of samples: {df.shape[0]}
Number of features: {df.shape[1]}
Features: {', '.join(df.columns.tolist())}
Target variable: {target_column if 'target_column' in locals() else 'Not selected'}
"""
    return context 