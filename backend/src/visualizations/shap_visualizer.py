"""
SHAP Explainability System
============================

Simple SHAP implementation for M5 Walmart forecasting models:
- Global feature importance analysis
- Local explanations for specific predictions
- Business-friendly visualizations
- Integration with existing dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
import warnings
from datetime import datetime, date

# SHAP and ML libraries
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Suppress SHAP warnings
warnings.filterwarnings("ignore", message=".*SHAP.*")

# -------------------------------
# SHAP Explainer Classes
# -------------------------------

class ShapExplainer:
    """Simple SHAP explainer for forecasting models."""
    
    def __init__(self, model, model_name: str = "lightgbm"):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (LightGBM, Prophet, etc.)
            model_name: Name of the model type
        """
        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        
        logger.info(f"Initialized SHAP explainer for {model_name}")
    
    def fit_explainer(self, background_data: pd.DataFrame, max_background_samples: int = 100):
        """
        Fit SHAP explainer with background data.
        
        Args:
            background_data: Representative sample of training data
            max_background_samples: Maximum samples to use for background
        """
        logger.info("Fitting SHAP explainer...")
        
        # Prepare background data
        if len(background_data) > max_background_samples:
            self.background_data = background_data.sample(n=max_background_samples, random_state=42)
        else:
            self.background_data = background_data.copy()
        
        # Get numeric features only
        numeric_cols = self.background_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['sales', 'actual_value']]
        
        self.feature_names = feature_cols
        background_features = self.background_data[feature_cols]
        
        # Create appropriate explainer based on model type
        if self.model_name == "lightgbm":
            # TreeExplainer for LightGBM
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Created TreeExplainer for LightGBM")
            
        else:
            # KernelExplainer for other models (slower but works with any model)
            def model_predict(X):
                return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(model_predict, background_features)
            logger.info("Created KernelExplainer for generic model")
        
        logger.info(f"SHAP explainer fitted with {len(background_features)} background samples")
    
    def explain_prediction(self, input_data: pd.DataFrame, max_samples: int = 10) -> shap.Explanation:
        """
        Generate SHAP explanations for input data.
        
        Args:
            input_data: Data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer() first.")
        
        # Prepare input features
        if len(input_data) > max_samples:
            explain_data = input_data.sample(n=max_samples, random_state=42)
        else:
            explain_data = input_data.copy()
        
        features_to_explain = explain_data[self.feature_names]
        
        logger.info(f"Generating SHAP explanations for {len(features_to_explain)} samples...")
        
        try:
            # Generate SHAP values
            if self.model_name == "lightgbm":
                shap_values = self.explainer.shap_values(features_to_explain)
                
                # Create explanation object
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=self.explainer.expected_value,
                    data=features_to_explain,
                    feature_names=self.feature_names
                )
            else:
                # For KernelExplainer
                explanation = self.explainer.shap_values(features_to_explain)
            
            logger.info("SHAP explanations generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            raise
    
    def get_global_importance(self, explanation: shap.Explanation, top_k: int = 20) -> pd.DataFrame:
        """
        Get global feature importance from SHAP explanations.
        
        Args:
            explanation: SHAP explanation object
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(explanation.values), axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)

# -------------------------------
# Visualization Functions
# -------------------------------

class ShapVisualizer:
    """Create business-friendly SHAP visualizations."""
    
    def __init__(self, explainer: ShapExplainer):
        self.explainer = explainer
    
    def plot_global_importance(self, explanation: shap.Explanation, 
                             top_k: int = 15, save_path: Optional[str] = None) -> go.Figure:
        """Create global feature importance plot."""
        importance_df = self.explainer.get_global_importance(explanation, top_k)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            y=importance_df['feature'][::-1],  # Reverse for descending order
            x=importance_df['importance'][::-1],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=f"Top {top_k} Most Important Features - {self.explainer.model_name}",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            height=600,
            margin=dict(l=200)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Global importance plot saved to {save_path}")
        
        return fig
    
    def plot_waterfall_explanation(self, explanation: shap.Explanation, 
                                  sample_idx: int = 0, 
                                  save_path: Optional[str] = None) -> str:
        """
        Create waterfall plot for single prediction explanation.
        
        Args:
            explanation: SHAP explanation object
            sample_idx: Index of sample to explain
            save_path: Path to save HTML plot
            
        Returns:
            HTML string of the plot
        """
        try:
            # Create matplotlib waterfall plot
            plt.figure(figsize=(10, 8))
            
            # Get SHAP values for the specific sample
            shap_values = explanation.values[sample_idx]
            base_value = explanation.base_values if hasattr(explanation, 'base_values') else 0
            feature_values = explanation.data.iloc[sample_idx] if hasattr(explanation, 'data') else None
            
            # Create waterfall plot using SHAP
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values, 
                    base_values=base_value,
                    data=feature_values,
                    feature_names=self.explainer.feature_names
                ),
                show=False
            )
            
            plt.title(f"Prediction Explanation - Sample {sample_idx}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.html', '.png'), dpi=150, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to {save_path.replace('.html', '.png')}")
            
            plt.close()
            
            return f"Waterfall explanation created for sample {sample_idx}"
            
        except Exception as e:
            logger.error(f"Failed to create waterfall plot: {e}")
            return f"Failed to create waterfall plot: {e}"
    
    def plot_summary_plot(self, explanation: shap.Explanation, 
                         max_display: int = 20,
                         save_path: Optional[str] = None) -> str:
        """Create SHAP summary plot."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Create summary plot
            shap.summary_plot(
                explanation.values,
                explanation.data if hasattr(explanation, 'data') else None,
                feature_names=self.explainer.feature_names,
                max_display=max_display,
                show=False
            )
            
            plt.title("SHAP Summary Plot - Feature Impact Distribution")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.html', '_summary.png'), dpi=150, bbox_inches='tight')
                logger.info(f"Summary plot saved to {save_path.replace('.html', '_summary.png')}")
            
            plt.close()
            
            return f"Summary plot created with {max_display} features"
            
        except Exception as e:
            logger.error(f"Failed to create summary plot: {e}")
            return f"Failed to create summary plot: {e}"
    
    def create_business_explanation(self, explanation: shap.Explanation, 
                                  sample_idx: int = 0,
                                  prediction_value: float = None,
                                  store_id: str = "Unknown",
                                  item_id: str = "Unknown",
                                  pred_date: str = "Unknown") -> Dict[str, Any]:
        """
        Create business-friendly explanation text.
        
        Args:
            explanation: SHAP explanation object
            sample_idx: Index of sample to explain
            prediction_value: Actual prediction value
            store_id: Store identifier
            item_id: Item identifier
            pred_date: Prediction date
            
        Returns:
            Dictionary with business explanation
        """
        try:
            shap_values = explanation.values[sample_idx]
            base_value = explanation.base_values if hasattr(explanation, 'base_values') else 0
            
            # Get top positive and negative contributors
            feature_impacts = list(zip(self.explainer.feature_names, shap_values))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_positive = [f for f in feature_impacts if f[1] > 0][:3]
            top_negative = [f for f in feature_impacts if f[1] < 0][:3]
            
            # Create business explanation
            explanation_text = {
                'prediction_info': {
                    'store_id': store_id,
                    'item_id': item_id,
                    'date': pred_date,
                    'predicted_sales': prediction_value,
                    'baseline_sales': base_value
                },
                'key_drivers': {
                    'increasing_sales': [
                        {
                            'feature': self._business_friendly_name(feature),
                            'impact': f"+{impact:.2f}",
                            'description': self._get_feature_description(feature, impact)
                        } for feature, impact in top_positive
                    ],
                    'decreasing_sales': [
                        {
                            'feature': self._business_friendly_name(feature),
                            'impact': f"{impact:.2f}",
                            'description': self._get_feature_description(feature, impact)
                        } for feature, impact in top_negative
                    ]
                },
                'summary': self._generate_summary(top_positive, top_negative, prediction_value, base_value)
            }
            
            return explanation_text
            
        except Exception as e:
            logger.error(f"Failed to create business explanation: {e}")
            return {"error": str(e)}
    
    def _business_friendly_name(self, feature_name: str) -> str:
        """Convert technical feature names to business-friendly names."""
        name_mapping = {
            'sales_lag_7': '7-day Historical Sales',
            'sales_lag_14': '14-day Historical Sales', 
            'sales_lag_28': '28-day Historical Sales',
            'sales_roll_7_mean': '7-day Average Sales',
            'sales_roll_14_mean': '14-day Average Sales',
            'sales_roll_28_mean': '28-day Average Sales',
            'price_lag_7': 'Recent Price',
            'price_vs_category': 'Price vs Category Average',
            'is_weekend': 'Weekend Effect',
            'is_holiday': 'Holiday Effect',
            'is_promo': 'Promotion Effect',
            'month': 'Seasonal Month Effect',
            'dayofweek': 'Day of Week Effect',
            'snap_any': 'SNAP Benefits Effect'
        }
        
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())
    
    def _get_feature_description(self, feature_name: str, impact: float) -> str:
        """Get business description for feature impact."""
        abs_impact = abs(impact)
        direction = "increases" if impact > 0 else "decreases"
        
        descriptions = {
            'sales_lag_7': f"Recent week's performance {direction} expected sales by {abs_impact:.1f} units",
            'sales_lag_14': f"Two weeks ago performance {direction} expected sales by {abs_impact:.1f} units",
            'price_lag_7': f"Current pricing strategy {direction} expected sales by {abs_impact:.1f} units",
            'is_weekend': f"Weekend shopping patterns {direction} expected sales by {abs_impact:.1f} units",
            'is_holiday': f"Holiday effect {direction} expected sales by {abs_impact:.1f} units",
            'is_promo': f"Promotional activity {direction} expected sales by {abs_impact:.1f} units",
            'snap_any': f"SNAP benefits availability {direction} expected sales by {abs_impact:.1f} units"
        }
        
        return descriptions.get(feature_name, f"This factor {direction} expected sales by {abs_impact:.1f} units")
    
    def _generate_summary(self, top_positive: List, top_negative: List, 
                         prediction: float, baseline: float) -> str:
        """Generate a business summary of the prediction."""
        if not top_positive and not top_negative:
            return "Prediction is close to the baseline with no major driving factors."
        
        summary = f"The model predicts {prediction:.1f} units (vs baseline of {baseline:.1f}). "
        
        if top_positive:
            main_driver = top_positive[0]
            summary += f"The main positive factor is {self._business_friendly_name(main_driver[0])} (+{main_driver[1]:.1f}). "
        
        if top_negative:
            main_detractor = top_negative[0]
            summary += f"The main negative factor is {self._business_friendly_name(main_detractor[0])} ({main_detractor[1]:.1f}). "
        
        return summary

# -------------------------------
# Main SHAP System
# -------------------------------

class ShapSystem:
    """Main system for M5 SHAP explanations."""
    
    def __init__(self, model_dir: str = "models", output_dir: str = "shap_explanations"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.explainers = {}
        self.visualizers = {}
        
        logger.info("M5 SHAP System initialized")
    
    def load_model_and_create_explainer(self, model_name: str, model_path: str = None) -> ShapExplainer:
        """Load model and create SHAP explainer."""
        if model_path is None:
            model_path = self.model_dir / f"{model_name}.pkl"
        
        logger.info(f"Loading model from {model_path}")
        
        try:
            if model_name == "lightgbm":
                model = lgb.Booster(model_file=str(model_path))
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            explainer = ShapExplainer(model, model_name)
            self.explainers[model_name] = explainer
            self.visualizers[model_name] = ShapVisualizer(explainer)
            
            logger.info(f"Created SHAP explainer for {model_name}")
            return explainer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def explain_predictions(self, model_name: str, input_data: pd.DataFrame,
                          background_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate complete SHAP explanations for predictions."""
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model {model_name}")
        
        explainer = self.explainers[model_name]
        visualizer = self.visualizers[model_name]
        
        # Fit explainer if background data provided
        if background_data is not None:
            explainer.fit_explainer(background_data)
        
        # Generate explanations
        explanation = explainer.explain_prediction(input_data)
        
        # Create visualizations
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'samples_explained': len(input_data),
            'explanations': {}
        }
        
        # Global importance
        global_fig = visualizer.plot_global_importance(
            explanation, 
            save_path=str(self.output_dir / f"{model_name}_global_importance.html")
        )
        results['global_importance_plot'] = f"{model_name}_global_importance.html"
        
        # Individual explanations for first few samples
        for i in range(min(3, len(input_data))):
            try:
                # Business explanation
                business_exp = visualizer.create_business_explanation(
                    explanation, 
                    sample_idx=i,
                    store_id=input_data.iloc[i].get('store_id', 'Unknown'),
                    item_id=input_data.iloc[i].get('item_id', 'Unknown'),
                    pred_date=str(input_data.iloc[i].get('date', 'Unknown'))
                )
                
                results['explanations'][f'sample_{i}'] = business_exp
                
                # Waterfall plot
                waterfall_result = visualizer.plot_waterfall_explanation(
                    explanation,
                    sample_idx=i,
                    save_path=str(self.output_dir / f"{model_name}_waterfall_{i}.html")
                )
                
            except Exception as e:
                logger.error(f"Failed to explain sample {i}: {e}")
        
        # Summary plot
        summary_result = visualizer.plot_summary_plot(
            explanation,
            save_path=str(self.output_dir / f"{model_name}_summary.html")
        )
        
        # Save results to JSON
        results_path = self.output_dir / f"{model_name}_explanations.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"SHAP explanations saved to {results_path}")
        
        return results

# -------------------------------
# CLI Interface
# -------------------------------

def main():
    """CLI interface for SHAP explanations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="M5 SHAP Explainability System")
    parser.add_argument("--model", required=True, help="Model name (lightgbm, prophet, etc.)")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--input-data", required=True, help="Input data CSV/parquet for explanations")
    parser.add_argument("--background-data", help="Background data CSV/parquet for fitting explainer")
    parser.add_argument("--output-dir", default="shap_explanations", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples to explain")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize SHAP system
        shap_system = ShapSystem(output_dir=args.output_dir)
        
        # Load model and create explainer
        explainer = shap_system.load_model_and_create_explainer(args.model, args.model_path)
        
        # Load input data
        logger.info(f"Loading input data from {args.input_data}")
        if args.input_data.endswith('.csv'):
            input_data = pd.read_csv(args.input_data)
        else:
            input_data = pd.read_parquet(args.input_data)
        
        # Limit samples
        if len(input_data) > args.max_samples:
            input_data = input_data.sample(n=args.max_samples, random_state=42)
        
        # Load background data if provided
        background_data = None
        if args.background_data:
            logger.info(f"Loading background data from {args.background_data}")
            if args.background_data.endswith('.csv'):
                background_data = pd.read_csv(args.background_data)
            else:
                background_data = pd.read_parquet(args.background_data)
        
        # Generate explanations
        results = shap_system.explain_predictions(args.model, input_data, background_data)
        
        print("SHAP explanations generated successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Samples explained: {results['samples_explained']}")
        
        # Print sample business explanation
        if 'sample_0' in results['explanations']:
            sample_exp = results['explanations']['sample_0']
            print("\nSample Business Explanation:")
            print(f"Prediction: {sample_exp['prediction_info']}")
            print(f"Summary: {sample_exp['summary']}")
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())