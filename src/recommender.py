from typing import List, Dict, Any
from .heuristics import MODEL_CATALOG, ModelInfo

class ModelRecommender:
    def __init__(self, dataset_characteristics: Dict[str, Any]):
        self.characteristics = dataset_characteristics
        
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Return a list of recommended models with explanations"""
        recommendations = []
        
        for model_id, model_info in MODEL_CATALOG.items():
            score, reasons = self._evaluate_model_fit(model_info)
            if score > 0:
                recommendations.append({
                    'model_id': model_id,
                    'model_info': model_info,
                    'score': score,
                    'reasons': reasons
                })
        
        # Sort by score in descending order
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def _evaluate_model_fit(self, model_info: ModelInfo) -> tuple[float, List[str]]:
        """Evaluate how well a model fits the dataset characteristics"""
        score = 1.0
        reasons = []
        
        # Check if model is suitable for the problem type
        if self.characteristics['target_type'] not in model_info.suitable_for:
            return 0, []
        
        # Check dataset size
        n_samples = self.characteristics['n_samples']
        if n_samples < model_info.min_samples:
            return 0, []
        if n_samples > model_info.max_samples:
            return 0, []
            
        # Handle dataset size preferences
        if n_samples < 1000:
            if model_info.complexity == 'low':
                score += 0.3
                reasons.append("Good fit for small datasets")
        elif n_samples > 10000:
            if model_info.complexity == 'high':
                score += 0.3
                reasons.append("Can handle large datasets effectively")
        
        # Check for missing values
        if self.characteristics['has_missing_values']:
            if model_info.handles_missing_values:
                score += 0.2
                reasons.append("Can handle missing values")
            else:
                score -= 0.2
                reasons.append("Requires preprocessing for missing values")
        
        # Check for categorical features
        if self.characteristics['has_categorical']:
            if model_info.handles_categorical:
                score += 0.2
                reasons.append("Can handle categorical features directly")
            else:
                score -= 0.1
                reasons.append("Requires encoding of categorical features")
        
        # Check for class imbalance
        if self.characteristics['is_imbalanced']:
            if model_info.handles_imbalanced:
                score += 0.2
                reasons.append("Suitable for imbalanced datasets")
            else:
                score -= 0.2
                reasons.append("May need additional techniques for handling imbalanced data")
        
        return score, reasons 