"""
CTR Prediction Model for Ad Copy Optimization

This module provides a Random Forest-based model to predict Click-Through Rate (CTR)
from engineered ad copy features. Designed for production use with proper preprocessing
and model persistence.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Union, Optional
import warnings
warnings.filterwarnings('ignore')


class CTRPredictor:
    """
    Random Forest-based CTR prediction model for ad copy optimization.
    
    Features:
    - Handles mixed data types (text, categorical, numerical)
    - Robust to outliers and missing values
    - Provides feature importance insights
    - Production-ready with model persistence
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.feature_importance_ = None
        self.training_metrics_ = None
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        Removes performance-derived features to prevent data leakage.
        
        Args:
            df: Input dataframe with engineered features
            
        Returns:
            Cleaned dataframe ready for modeling
        """
        # Remove target variables and IDs
        exclude_cols = ['ad_id', 'ctr', 'high_performer']
        
        # Remove performance-derived features that cause data leakage
        performance_features = [
            'clicks', 'impressions', 'conversions', 'spend',  # Raw performance metrics
            'cpc', 'cpa', 'cvr',  # Calculated performance metrics
            'cost_per_thousand_impressions', 'revenue_efficiency',  # Cost metrics
            'engagement_score'  # Derived engagement metrics
        ]
        
        # Also remove interaction features that use performance metrics
        interaction_patterns = ['_x_ctr', '_x_cvr', 'clicks_x_', 'impressions_x_', 
                              'spend_x_', 'conversions_x_']
        
        exclude_cols.extend(performance_features)
        
        # Find and exclude interaction features with performance metrics
        for col in df.columns:
            if any(pattern in col for pattern in interaction_patterns):
                exclude_cols.append(col)
        
        # Keep only valid predictive features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"🔒 Excluded {len(exclude_cols)} features to prevent data leakage")
        print(f"📝 Using {len(feature_cols)} clean predictive features")
        
        # Handle missing values (fill with 0 for ad copy features)
        df_clean = df[feature_cols].fillna(0)
        
        # Store feature columns for prediction consistency
        if self.feature_columns is None:
            self.feature_columns = feature_cols
            
        return df_clean
    
    def train(self, df: pd.DataFrame, target_col: str = 'ctr', 
              test_size: float = 0.2, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the CTR prediction model.
        
        Args:
            df: Training dataframe with engineered features and CTR target
            target_col: Name of target column (default: 'ctr')
            test_size: Fraction of data for testing (default: 0.2)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training metrics and model info
        """
        print("🚀 Training CTR Prediction Model...")
        
        # Prepare features and target
        X = self._prepare_features(df)
        y = df[target_col]
        
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        if tune_hyperparameters:
            print("🔧 Tuning hyperparameters...")
            # Grid search for optimal parameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestRegressor(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters optimized for ad data
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store metrics
        self.training_metrics_ = {
            'train': train_metrics,
            'test': test_metrics,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        self.is_trained = True
        
        # Print results
        print(f"\\n📈 Model Performance:")
        print(f"   Train R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.6f}")
        print(f"   Test R²:  {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.6f}")
        
        # Check for potential overfitting
        self._validate_model_performance(train_metrics, test_metrics)
        
        return self.training_metrics_
    
    def _validate_model_performance(self, train_metrics: Dict, test_metrics: Dict) -> None:
        """
        Validate model performance and warn about potential issues.
        
        Args:
            train_metrics: Training set performance metrics
            test_metrics: Test set performance metrics
        """
        train_r2 = train_metrics['r2']
        test_r2 = test_metrics['r2']
        test_rmse = test_metrics['rmse']
        
        # Check for overfitting
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.1:  # More than 10% gap
            print(f"\n⚠️  WARNING: Possible overfitting detected!")
            print(f"   R² gap (train-test): {r2_gap:.4f}")
            print(f"   Consider: reducing model complexity, more data, or better regularization")
        
        # Check for unrealistic performance (data leakage)
        if train_r2 > 0.95 or test_rmse < 0.001:
            print(f"\n🚨 ALERT: Unrealistic performance - possible data leakage!")
            print(f"   Train R²: {train_r2:.4f} (>95% suggests leakage)")
            print(f"   Test RMSE: {test_rmse:.6f} (<0.001 suggests leakage)")
            print(f"   Real-world CTR prediction typically achieves R² of 0.15-0.40")
        
        # Provide realistic expectations
        if test_r2 > 0.4:
            print(f"\n🎯 Performance Assessment: Excellent (R² > 0.4)")
        elif test_r2 > 0.25:
            print(f"\n🎯 Performance Assessment: Good (R² > 0.25)")
        elif test_r2 > 0.15:
            print(f"\n🎯 Performance Assessment: Acceptable (R² > 0.15)")
        else:
            print(f"\n🎯 Performance Assessment: Needs improvement (R² < 0.15)")
    
    def predict_ctr_features(self, ad_row: Union[pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Predict CTR for given ad features.
        
        Args:
            ad_row: Single ad row (Series) or multiple ads (DataFrame) with engineered features
            
        Returns:
            Predicted CTR value(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Handle single row prediction
        if isinstance(ad_row, pd.Series):
            ad_row = pd.DataFrame([ad_row])
        
        # Prepare features (ensure same columns as training, add missing columns with 0)
        X_input = pd.DataFrame(index=ad_row.index)
        
        # Add all required feature columns, filling missing ones with 0
        for col in self.feature_columns:
            if col in ad_row.columns:
                X_input[col] = ad_row[col]
            else:
                X_input[col] = 0
                
        X = X_input.fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Return single value for single prediction
        if len(predictions) == 1:
            return float(predictions[0])
        
        return predictions
    
    def evaluate_ad_improvements(self, improved_ads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate predicted CTR improvements from generator output.
        
        Args:
            improved_ads_df: DataFrame with engineered features from generator (model-ready format)
                           Should contain both original and improved versions with '_original' and '_improved' suffixes
            
        Returns:
            DataFrame with evaluation results comparing original vs improved predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        print("🔍 Evaluating Ad Copy Improvements...")
        
        # Check if ad_id column exists, if not create it from index
        if 'ad_id' not in improved_ads_df.columns:
            print("⚠️  ad_id column missing, creating from row indices...")
            # Assume first half are original, second half are improved
            n_ads = len(improved_ads_df) // 2
            ad_ids = []
            for i in range(n_ads):
                ad_ids.append(f"ad_{i:04d}_original")
            for i in range(n_ads):
                ad_ids.append(f"ad_{i:04d}_improved")
            improved_ads_df = improved_ads_df.copy()
            improved_ads_df['ad_id'] = ad_ids
        
        # Separate original and improved versions
        original_ads = improved_ads_df[improved_ads_df['ad_id'].str.endswith('_original')].copy()
        improved_ads = improved_ads_df[improved_ads_df['ad_id'].str.endswith('_improved')].copy()
        
        if len(original_ads) == 0 or len(improved_ads) == 0:
            raise ValueError("Dataset must contain both '_original' and '_improved' ad versions")
        
        print(f"📊 Evaluating {len(original_ads)} ad pairs...")
        
        # Predict CTR for both versions
        original_predictions = self.predict_ctr_features(original_ads)
        improved_predictions = self.predict_ctr_features(improved_ads)
        
        # Create comparison results
        results = []
        
        for i, (orig_idx, orig_row) in enumerate(original_ads.iterrows()):
            # Find corresponding improved version
            base_ad_id = orig_row['ad_id'].replace('_original', '')
            improved_row = improved_ads[improved_ads['ad_id'] == f"{base_ad_id}_improved"].iloc[0]
            
            orig_ctr = original_predictions[i] if isinstance(original_predictions, np.ndarray) else original_predictions
            improved_ctr = improved_predictions[i] if isinstance(improved_predictions, np.ndarray) else improved_predictions
            
            # Calculate improvement metrics
            ctr_improvement = improved_ctr - orig_ctr
            ctr_improvement_pct = (ctr_improvement / orig_ctr * 100) if orig_ctr > 0 else 0
            
            result = {
                'ad_id': base_ad_id,
                'original_headline': getattr(orig_row, 'headline_text', ''),
                'improved_headline': getattr(improved_row, 'headline_text', ''),
                'original_body': getattr(orig_row, 'body_text', ''),
                'improved_body': getattr(improved_row, 'body_text', ''),
                'predicted_ctr_original': float(orig_ctr),
                'predicted_ctr_improved': float(improved_ctr),
                'ctr_improvement_abs': float(ctr_improvement),
                'ctr_improvement_pct': float(ctr_improvement_pct),
                'improvement_positive': ctr_improvement > 0,
                'category': getattr(orig_row, 'category', 'Unknown'),
                'platform': getattr(orig_row, 'platform', 'Unknown')
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        self._print_evaluation_summary(results_df)
        
        return results_df
    
    def _print_evaluation_summary(self, results_df: pd.DataFrame) -> None:
        """
        Print evaluation summary statistics.
        
        Args:
            results_df: DataFrame with evaluation results
        """
        total_ads = len(results_df)
        improved_count = results_df['improvement_positive'].sum()
        improvement_rate = (improved_count / total_ads * 100) if total_ads > 0 else 0
        
        avg_original_ctr = results_df['predicted_ctr_original'].mean()
        avg_improved_ctr = results_df['predicted_ctr_improved'].mean()
        avg_improvement_abs = results_df['ctr_improvement_abs'].mean()
        avg_improvement_pct = results_df['ctr_improvement_pct'].mean()
        
        print(f"\\n📈 Ad Copy Improvement Evaluation Results:")
        print(f"   Total ad pairs evaluated: {total_ads}")
        print(f"   Ads with positive improvement: {improved_count} ({improvement_rate:.1f}%)")
        print(f"   Average original CTR: {avg_original_ctr:.4f} ({avg_original_ctr*100:.2f}%)")
        print(f"   Average improved CTR: {avg_improved_ctr:.4f} ({avg_improved_ctr*100:.2f}%)")
        print(f"   Average absolute improvement: {avg_improvement_abs:.4f} ({avg_improvement_abs*100:.2f} percentage points)")
        print(f"   Average relative improvement: {avg_improvement_pct:.1f}%")
        
        # Top improvements
        top_improvements = results_df.nlargest(3, 'ctr_improvement_pct')[['ad_id', 'ctr_improvement_pct']]
        print(f"\\n🏆 Top 3 Improvements:")
        for _, row in top_improvements.iterrows():
            print(f"   {row['ad_id']}: +{row['ctr_improvement_pct']:.1f}%")
        
        # Category breakdown
        if 'category' in results_df.columns:
            category_stats = results_df.groupby('category').agg({
                'improvement_positive': 'sum',
                'ctr_improvement_pct': 'mean'
            }).round(2)
            print(f"\\n📊 Performance by Category:")
            for category, stats in category_stats.iterrows():
                print(f"   {category}: {stats['improvement_positive']} improved, avg +{stats['ctr_improvement_pct']:.1f}%")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first.")
            
        return self.feature_importance_.head(top_n)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
            
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance_,
            'training_metrics': self.training_metrics_
        }, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance_ = model_data['feature_importance']
        self.training_metrics_ = model_data['training_metrics']
        self.is_trained = True
        
        print(f"📁 Model loaded from: {filepath}")


def train_ctr_model(csv_path: str, model_save_path: str = None, 
                   tune_hyperparameters: bool = True) -> CTRPredictor:
    """
    Convenience function to train CTR model from CSV.
    
    Args:
        csv_path: Path to engineered features CSV
        model_save_path: Optional path to save trained model
        tune_hyperparameters: Whether to tune hyperparameters
        
    Returns:
        Trained CTRPredictor instance
    """
    # Load data
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Initialize and train model
    predictor = CTRPredictor()
    predictor.train(df, tune_hyperparameters=tune_hyperparameters)
    
    # Save model if requested
    if model_save_path:
        predictor.save_model(model_save_path)
    
    # Print feature importance insights
    print(f"\\n🎯 Top 10 Most Important Features (Clean - No Data Leakage):")
    top_features = predictor.get_feature_importance(10)
    for idx, row in top_features.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        
        # Add text label for feature type
        if any(keyword in feature_name for keyword in ['len', 'word_count', 'caps', 'numbers']):
            feature_type = "[TEXT]"
        elif any(keyword in feature_name for keyword in ['platform', 'format', 'category']):
            feature_type = "[CATEGORICAL]"
        elif any(keyword in feature_name for keyword in ['urgency', 'positive', 'sentiment']):
            feature_type = "[CONTENT]"
        else:
            feature_type = "[ENGINEERED]"
            
        print(f"   {feature_type:12} {feature_name[:30]:30} : {importance:.4f}")
    
    return predictor


def evaluate_improved_ads(model_path: str, improved_ads_csv: str, 
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to evaluate improved ad copies using trained model.
    
    Args:
        model_path: Path to trained CTR prediction model
        improved_ads_csv: Path to CSV with model-ready improved ads (from generator)
        output_path: Optional path to save evaluation results
        
    Returns:
        DataFrame with evaluation results
    """
    # Load trained model
    print(f"📁 Loading model from: {model_path}")
    predictor = CTRPredictor()
    predictor.load_model(model_path)
    
    # Load improved ads data
    print(f"📂 Loading improved ads from: {improved_ads_csv}")
    improved_ads_df = pd.read_csv(improved_ads_csv)
    
    # Evaluate improvements
    results = predictor.evaluate_ad_improvements(improved_ads_df)
    
    # Save results if requested
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"💾 Evaluation results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    csv_path = "../../data/ml_ready_ad_features.csv"
    model_path = "../../models/ctr_predictor.joblib"
    
    # Train model
    predictor = train_ctr_model(csv_path, model_path)
    
    print(f"\\n✅ CTR prediction model ready!")
    print(f"📞 Use predictor.predict_ctr_features(ad_row) to make predictions")
    print(f"📊 Use evaluate_improved_ads(model_path, improved_ads_csv) to evaluate improvements")
