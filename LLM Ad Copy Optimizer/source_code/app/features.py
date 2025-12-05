"""
Feature Engineering Pipeline for Ad Copy Data

This module provides functionality to engineer features from raw ad copy data,
including text analysis, categorical encoding, and performance metrics preparation.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List
import os


class AdCopyFeatureEngineer:
    """
    A comprehensive feature engineering pipeline for ad copy data.
    
    Handles text features, categorical encoding, and performance metrics
    to prepare data for machine learning models.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def drop_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are completely empty (all null values).
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with empty columns removed
        """
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            print(f"Dropping empty columns: {empty_cols}")
            df = df.drop(columns=empty_cols)
        return df
    
    def engineer_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate text-based features from headline and body text.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional text features
        """
        # Handle NaN values in text columns
        df['headline_text'] = df['headline_text'].fillna('')
        df['body_text'] = df['body_text'].fillna('')
        df['call_to_action'] = df['call_to_action'].fillna('')
        
        # Basic length features
        df['headline_len'] = df['headline_text'].str.len()
        df['body_len'] = df['body_text'].str.len()
        df['cta_len'] = df['call_to_action'].str.len()
        
        # Word count features
        df['headline_word_count'] = df['headline_text'].str.split().str.len()
        df['body_word_count'] = df['body_text'].str.split().str.len()
        df['cta_word_count'] = df['call_to_action'].str.split().str.len()
        
        # Text quality features
        df['headline_exclamation'] = df['headline_text'].str.count('!').fillna(0).astype(int)
        df['headline_question'] = df['headline_text'].str.count(r'\?').fillna(0).astype(int)
        df['body_exclamation'] = df['body_text'].str.count('!').fillna(0).astype(int)
        df['body_question'] = df['body_text'].str.count(r'\?').fillna(0).astype(int)
        
        # Capitalization features
        df['headline_caps_ratio'] = df['headline_text'].apply(self._calculate_caps_ratio)
        df['body_caps_ratio'] = df['body_text'].apply(self._calculate_caps_ratio)
        
        # Number presence
        df['headline_has_numbers'] = df['headline_text'].str.contains(r'\d').fillna(False).astype(int)
        df['body_has_numbers'] = df['body_text'].str.contains(r'\d').fillna(False).astype(int)
        
        # Percentage/discount detection
        df['headline_has_percent'] = df['headline_text'].str.contains(r'%|\bpercent\b', case=False).fillna(False).astype(int)
        df['body_has_percent'] = df['body_text'].str.contains(r'%|\bpercent\b', case=False).fillna(False).astype(int)
        
        # Urgency words
        urgency_words = r'\b(limited|hurry|now|today|urgent|quick|fast|instant|immediate)\b'
        df['headline_urgency'] = df['headline_text'].str.contains(urgency_words, case=False).fillna(False).astype(int)
        df['body_urgency'] = df['body_text'].str.contains(urgency_words, case=False).fillna(False).astype(int)
        
        # Sentiment indicators (basic)
        positive_words = r'\b(free|save|best|great|amazing|awesome|easy|simple)\b'
        df['headline_positive'] = df['headline_text'].str.contains(positive_words, case=False).fillna(False).astype(int)
        df['body_positive'] = df['body_text'].str.contains(positive_words, case=False).fillna(False).astype(int)
        
        return df
    
    def _calculate_caps_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters to total letters."""
        if pd.isna(text) or len(text) == 0:
            return 0.0
        letters = re.sub(r'[^a-zA-Z]', '', text)
        if len(letters) == 0:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)
    
    def engineer_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded categorical features
        """
        categorical_cols = ['platform', 'ad_format', 'category', 'call_to_action']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create label encoded version
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                
                # Create one-hot encoded versions for important categories
                if col in ['platform', 'ad_format', 'category']:
                    dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
                    df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create core and additional performance-based features.
        Includes CTR, CVR, CPC, CPA, high_performer if not already present.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with performance features
        """
        # Fill NaN values to avoid division issues
        df["impressions"] = df["impressions"].fillna(0)
        df["clicks"] = df["clicks"].fillna(0)
        df["conversions"] = df["conversions"].fillna(0)
        df["spend"] = df["spend"].fillna(0.0)
        
        # Core performance metrics (if not already present)
        if 'ctr' not in df.columns:
            df["ctr"] = df.apply(
                lambda row: row["clicks"] / row["impressions"] if row["impressions"] > 0 else 0.0,
                axis=1,
            )
        
        if 'cvr' not in df.columns:
            df["cvr"] = df.apply(
                lambda row: row["conversions"] / row["clicks"] if row["clicks"] > 0 else 0.0,
                axis=1,
            )
        
        if 'cpc' not in df.columns:
            df["cpc"] = df.apply(
                lambda row: row["spend"] / row["clicks"] if row["clicks"] > 0 else 0.0,
                axis=1,
            )
        
        if 'cpa' not in df.columns:
            df["cpa"] = df.apply(
                lambda row: row["spend"] / row["conversions"] if row["conversions"] > 0 else 0.0,
                axis=1,
            )
        
        if 'high_performer' not in df.columns:
            # High performer flag using CTR 75th percentile threshold
            ctr_threshold = df["ctr"].quantile(0.75)
            df["high_performer"] = (df["ctr"] >= ctr_threshold).astype(int)
        
        # Additional engineered metrics
        df['revenue_efficiency'] = df['conversions'] / df['spend'].replace(0, np.nan)
        
        # Performance bins (handle cases where all values are the same)
        try:
            df['ctr_quartile'] = pd.qcut(df['ctr'], q=4, labels=['low', 'med_low', 'med_high', 'high'], duplicates='drop')
        except ValueError:
            # If all values are the same, assign all to 'med_low'
            df['ctr_quartile'] = 'med_low'
            
        try:
            df['cvr_quartile'] = pd.qcut(df['cvr'], q=4, labels=['low', 'med_low', 'med_high', 'high'], duplicates='drop')
        except ValueError:
            # If all values are the same, assign all to 'med_low'
            df['cvr_quartile'] = 'med_low'
        
        # Cost effectiveness
        df['cost_per_thousand_impressions'] = (df['spend'] / df['impressions']) * 1000
        
        # Engagement level
        df['engagement_score'] = (df['ctr'] * 0.6) + (df['cvr'] * 0.4)  # Weighted engagement
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        # Text length interactions with performance
        df['headline_len_x_ctr'] = df['headline_len'] * df['ctr']
        df['body_len_x_cvr'] = df['body_len'] * df['cvr']
        
        # Platform-format interactions (numeric encoding instead of strings)
        df['platform_facebook'] = (df['platform'] == 'facebook').astype(int)
        df['platform_instagram'] = (df['platform'] == 'instagram').astype(int)
        df['format_image'] = (df['ad_format'] == 'image').astype(int)
        df['format_video'] = (df['ad_format'] == 'video').astype(int)
        df['platform_format_interaction'] = (df['platform_facebook'] * df['format_video']) + (df['platform_instagram'] * df['format_image'])
        
        # Category-urgency interactions (numeric)
        df['total_urgency_score'] = df['headline_urgency'] + df['body_urgency']
        df['urgency_level'] = pd.cut(df['total_urgency_score'], bins=3, labels=False).fillna(0)
        
        return df
    
    def select_final_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most relevant features for machine learning.
        
        Args:
            df: Fully engineered dataframe
            
        Returns:
            Tuple of (final dataframe, list of selected feature names)
        """
        # Core identification - keep ad_id as string for tracking
        id_features = ['ad_id']
        
        # Text features
        text_features = [
            'headline_len', 'body_len', 'cta_len',
            'headline_word_count', 'body_word_count', 'cta_word_count',
            'headline_caps_ratio', 'body_caps_ratio',
            'headline_has_numbers', 'body_has_numbers',
            'headline_has_percent', 'body_has_percent',
            'headline_urgency', 'body_urgency',
            'headline_positive', 'body_positive',
            'headline_exclamation', 'headline_question',
            'body_exclamation', 'body_question'
        ]
        
        # Categorical features (encoded)
        categorical_features = [
            'platform_encoded', 'ad_format_encoded', 'category_encoded',
            'call_to_action_encoded'
        ]
        
        # One-hot encoded features
        onehot_features = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in ['platform_', 'ad_format_', 'category_']
        ) and col not in categorical_features]
        
        # Performance metrics
        performance_features = [
            'impressions', 'clicks', 'conversions', 'spend',
            'ctr', 'cvr', 'cpc', 'cpa'
        ]
        
        # Engineered performance features
        engineered_performance = [
            'revenue_efficiency', 'cost_per_thousand_impressions', 'engagement_score'
        ]
        
        # Interaction features (only numeric ones)
        interaction_features = [
            'headline_len_x_ctr', 'body_len_x_cvr', 
            'platform_format_interaction', 'total_urgency_score', 'urgency_level'
        ]
        
        # Target variable
        target_features = ['high_performer']
        
        # Combine all selected features
        selected_features = (
            id_features + text_features + categorical_features + 
            onehot_features + performance_features + 
            engineered_performance + interaction_features + target_features
        )
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        
        # Ensure all selected features are numeric (exclude any remaining string columns)
        df_final = df[available_features]
        
        # Check for and remove any remaining non-numeric columns (except ad_id)
        non_numeric_cols = df_final.select_dtypes(include=['object']).columns.tolist()
        # Keep ad_id even though it's non-numeric (we need it for evaluation)
        non_numeric_cols = [col for col in non_numeric_cols if col != 'ad_id']
        if non_numeric_cols:
            print(f"⚠️  Removing non-numeric columns: {non_numeric_cols}")
            df_final = df_final.drop(columns=non_numeric_cols)
            available_features = [f for f in available_features if f not in non_numeric_cols]
        
        return df_final, available_features
    
    def process_csv(self, input_path: str, output_path: str) -> None:
        """
        Complete feature engineering pipeline for a CSV file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save engineered features CSV
        """
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        
        print(f"Original shape: {df.shape}")
        
        # Step 1: Drop empty columns
        df = self.drop_empty_columns(df)
        print(f"After dropping empty columns: {df.shape}")
        
        # Step 2: Text feature engineering
        print("Engineering text features...")
        df = self.engineer_text_features(df)
        
        # Step 3: Categorical feature engineering
        print("Engineering categorical features...")
        df = self.engineer_categorical_features(df)
        
        # Step 4: Performance feature engineering
        print("Engineering performance features...")
        df = self.engineer_performance_features(df)
        
        # Step 5: Interaction features
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Step 6: Feature selection
        print("Selecting final features...")
        df_final, selected_features = self.select_final_features(df)
        
        # Save results
        print(f"Saving engineered features to: {output_path}")
        df_final.to_csv(output_path, index=False)
        
        print(f"Final shape: {df_final.shape}")
        # print(f"Selected features ({len(selected_features)}): {selected_features}")
        
        return df_final, selected_features


def engineer_features(input_csv_path: str, output_csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function for feature engineering in notebooks.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to save engineered features CSV
        
    Returns:
        Tuple of (engineered dataframe, list of feature names)
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run feature engineering
    engineer = AdCopyFeatureEngineer()
    return engineer.process_csv(input_csv_path, output_csv_path)
