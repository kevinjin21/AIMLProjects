"""
Ad Copy Generator for CTR Optimization

This module provides functionality to generate improved ad copy using LLM services
(Anthropic Claude or local Ollama) based on ML model insights and best practices.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this script
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables only.")

# Optional imports - will handle gracefully if not available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    
try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Import feature engineering for model-ready output
try:
    from .features import AdCopyFeatureEngineer
    FEATURES_AVAILABLE = True
except ImportError:
    try:
        from features import AdCopyFeatureEngineer
        FEATURES_AVAILABLE = True
    except ImportError:
        FEATURES_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log feature engineering availability
if not FEATURES_AVAILABLE:
    logger.warning("Feature engineering module not available. Some functionality will be limited.")

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class GeneratedAd(BaseModel):
    """Structured output for generated ad copy"""
    headline_text: str = Field(description="Optimized headline text")
    body_text: str = Field(description="Optimized body text")
    improvement_reasoning: str = Field(description="Brief explanation of improvements made")

@dataclass
class AdInsights:
    """Container for ML model insights to guide generation"""
    category: str
    platform: str
    ad_format: str
    current_ctr: float
    high_performer_features: Dict[str, float]
    optimization_suggestions: List[str]

class AdCopyGenerator:
    """
    AI-powered ad copy generator that improves CTR based on ML model insights.
    
    Supports both Anthropic Claude API and local Ollama models.
    Uses feature analysis from trained CTR prediction model to guide optimization.
    """
    
    def __init__(self, provider: LLMProvider = LLMProvider.ANTHROPIC, 
                 model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the generator with specified LLM provider.
        
        Args:
            provider: LLM provider (anthropic or ollama)
            model_name: Model name (claude-haiku-4-5 for Anthropic, llama3:8b for Ollama)
            api_key: API key for Anthropic (or set ANTHROPIC_API_KEY env var)
        """
        self.provider = provider
        
        if provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
            
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = model_name or "claude-haiku-4-5"
            
        elif provider == LLMProvider.OLLAMA:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama dependencies not installed. Run: pip install langchain-ollama")
            
            self.model_name = model_name or "llama3:8b"
            try:
                self.client = OllamaLLM(model=self.model_name)
                # Test connection
                self.client.invoke("test")
            except Exception as e:
                raise ConnectionError(f"Cannot connect to Ollama. Ensure it's running: {e}")
        
        # Setup output parser for structured responses
        if OLLAMA_AVAILABLE:
            self.output_parser = PydanticOutputParser(pydantic_object=GeneratedAd)
    
    def _create_optimization_prompt(self, headline: str, body: str, insights: Optional[AdInsights] = None) -> str:
        """
        Create a detailed prompt for ad copy optimization based on ML insights.
        
        Args:
            headline: Current headline text
            body: Current body text  
            insights: ML model insights for optimization guidance
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""You are an expert ad copywriter specializing in click-through rate (CTR) optimization. 

CURRENT AD COPY:
Headline: "{headline}"
Body: "{body}"

TASK: Rewrite this ad to significantly improve its CTR while maintaining the core offer and message authenticity.

OPTIMIZATION PRINCIPLES (BASED ON PERFORMANCE DATA):
1. LENGTH STRATEGY: If original headline is under 30 characters, expand to 30-35 characters with specific benefits. If over 40 characters, trim to 35-40 characters while preserving key value propositions.
2. BENEFIT-FOCUSED: Prioritize clear, specific value propositions over generic power words. Make benefits concrete and measurable.
3. CONTEXT-APPROPRIATE URGENCY: Use urgency words ("Limited", "Today", "Now") only when they naturally fit the offer context, not as generic additions.
4. SPECIFICITY OVER EMOTION: Include specific numbers, percentages, and outcomes rather than vague emotional appeals.
5. CLARITY FIRST: Ensure the value proposition is immediately clear. Avoid over-optimization that hurts authenticity.
6. COHESIVE MESSAGING: Ensure headline and body work together to reinforce the core benefit.
7. ACTIVE LANGUAGE: Use action-oriented, benefit-driven language that tells users exactly what they'll get.
8. ORIGINAL ASSESSMENT: Consider the original headline's length and clarity level to determine the best improvement strategy."""

        # Add transformation strategy guidance
        headline_length = len(headline)
        transformation_guidance = f"""

TRANSFORMATION STRATEGY (Current headline: {headline_length} chars):
        """
        
        if headline_length < 25:
            transformation_guidance += "\n• EXPAND: Add specific benefits, outcomes, or value propositions to reach 30-35 characters\n• FOCUS: Make the core value more explicit and concrete"
        elif headline_length < 30:
            transformation_guidance += "\n• ENHANCE: Add 5-10 characters with specific benefits or quantifiable outcomes\n• STRENGTHEN: Make value propositions more concrete and measurable"
        elif headline_length > 40:
            transformation_guidance += "\n• CONDENSE: Trim to 35-40 characters while preserving key value propositions\n• CLARIFY: Simplify language while maintaining benefit clarity"
        else:
            transformation_guidance += "\n• OPTIMIZE: Fine-tune for clarity and benefit focus within current length range\n• ENHANCE: Strengthen specific value propositions and outcomes"
            
        base_prompt += transformation_guidance

        # Add ML model insights if available
        if insights:
            insights_section = f"""

ML MODEL INSIGHTS:
- Category: {insights.category}
- Platform: {insights.platform}  
- Ad Format: {insights.ad_format}
- Current CTR: {insights.current_ctr:.4f} ({insights.current_ctr*100:.2f}%)
- High-performing features for this category: {', '.join([f"{k}: {v:.3f}" for k, v in insights.high_performer_features.items()])}

SPECIFIC RECOMMENDATIONS:
{chr(10).join([f"• {suggestion}" for suggestion in insights.optimization_suggestions])}"""
            base_prompt += insights_section

        # Format instructions
        format_instructions = """

RESPONSE FORMAT - CRITICAL:
You must respond with ONLY valid JSON. No explanations, no additional text, just pure JSON.

Expected structure:
{
  "headline_text": "Your optimized headline here",
  "body_text": "Your optimized body text here", 
  "improvement_reasoning": "Brief explanation of key improvements made"
}

CONSTRAINTS: 
- Do not invent new discounts, offers, or claims not in the original
- Preserve the core value proposition and maintain authenticity
- Focus on clarity and specific benefits over generic psychological triggers
- Avoid over-optimization - make natural improvements that enhance rather than replace the message
- Ensure claims remain truthful and substantiated
- Return ONLY the JSON object, nothing else

SUCCESS PATTERNS TO FOLLOW:
- Best performers expanded short headlines (+6 chars avg) with specific benefits
- Focus on concrete value propositions rather than generic urgency words
- Successful ads made the outcome/benefit more explicit and measurable"""

        return base_prompt + format_instructions
    
    def _extract_ad_insights(self, ad_data: pd.Series, feature_importance: Optional[pd.DataFrame] = None) -> AdInsights:
        """
        Extract ML model insights from ad feature data to guide optimization.
        
        Args:
            ad_data: Single row of ML feature data
            feature_importance: Feature importance from trained model
            
        Returns:
            AdInsights object with optimization guidance
        """
        # Extract basic info
        category = getattr(ad_data, 'category', 'Unknown')
        platform = getattr(ad_data, 'platform', 'Unknown') 
        ad_format = getattr(ad_data, 'ad_format', 'Unknown')
        current_ctr = getattr(ad_data, 'ctr', 0.0)
        
        # Identify high-performing features for this ad type
        high_performer_features = {}
        if feature_importance is not None:
            # Get top 5 most important features
            top_features = feature_importance.head(5)
            for _, row in top_features.iterrows():
                feature_name = row['feature']
                importance = row['importance']
                if feature_name in ad_data.index:
                    high_performer_features[feature_name] = importance
        
        # Generate optimization suggestions based on current features
        suggestions = []
        
        # Text length optimizations
        if hasattr(ad_data, 'headline_len'):
            if ad_data.headline_len < 20:
                suggestions.append("Consider lengthening headline to 25-40 characters for better engagement")
            elif ad_data.headline_len > 60:
                suggestions.append("Consider shortening headline for better mobile readability")
        
        # Content suggestions
        if hasattr(ad_data, 'headline_urgency') and ad_data.headline_urgency == 0:
            suggestions.append("Add urgency words like 'Limited Time', 'Now', 'Today' to increase CTR")
            
        if hasattr(ad_data, 'headline_has_numbers') and ad_data.headline_has_numbers == 0:
            suggestions.append("Consider adding specific numbers or percentages to the headline")
            
        if hasattr(ad_data, 'headline_positive') and ad_data.headline_positive == 0:
            suggestions.append("Include positive sentiment words to improve emotional appeal")
        
        # Platform-specific suggestions
        if platform == 'instagram' and ad_format == 'image':
            suggestions.append("Use visual-first language that complements image content")
        elif platform == 'facebook' and ad_format == 'video':
            suggestions.append("Create intrigue to encourage video play and engagement")
            
        # Category-specific suggestions
        if category == 'ecommerce':
            suggestions.append("Emphasize value proposition and clear call-to-action")
        elif category == 'education':
            suggestions.append("Focus on learning outcomes and transformation benefits")
        elif category == 'entertainment':
            suggestions.append("Use emotional hooks and curiosity to drive engagement")
        
        return AdInsights(
            category=category,
            platform=platform,
            ad_format=ad_format, 
            current_ctr=current_ctr,
            high_performer_features=high_performer_features,
            optimization_suggestions=suggestions
        )
    
    def generate_improved_copy(self, headline: str, body: str, 
                             ad_data: Optional[pd.Series] = None,
                             feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Generate improved ad copy using AI optimization.
        
        Args:
            headline: Current headline text
            body: Current body text
            ad_data: Optional feature data for ML insights
            feature_importance: Optional feature importance from trained model
            
        Returns:
            Dictionary with optimized headline, body, and reasoning
        """
        # Extract insights if data provided
        insights = None
        if ad_data is not None:
            insights = self._extract_ad_insights(ad_data, feature_importance)
        
        # Create optimization prompt
        prompt = self._create_optimization_prompt(headline, body, insights)
        
        try:
            if self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                
            elif self.provider == LLMProvider.OLLAMA:
                response = self.client.invoke(prompt)
                content = response
            
            # Parse JSON response - Claude often wraps JSON in text
            try:
                # First try direct JSON parsing
                result = json.loads(content)
                return {
                    'headline_text': result.get('headline_text', headline),
                    'body_text': result.get('body_text', body),
                    'improvement_reasoning': result.get('improvement_reasoning', 'No reasoning provided')
                }
            except json.JSONDecodeError:
                # Try to extract JSON from text response
                try:
                    # Look for JSON block in the response
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, content, re.DOTALL)
                    
                    for match in matches:
                        try:
                            result = json.loads(match)
                            if 'headline_text' in result and 'body_text' in result:
                                return {
                                    'headline_text': result.get('headline_text', headline),
                                    'body_text': result.get('body_text', body),
                                    'improvement_reasoning': result.get('improvement_reasoning', 'No reasoning provided')
                                }
                        except json.JSONDecodeError:
                            continue
                    
                    # If no valid JSON found, try to parse as text response
                    logger.info(f"Could not parse JSON, got response: {content[:200]}...")
                    return self._parse_text_response(content, headline, body)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse response: {e}")
                    return {
                        'headline_text': headline,
                        'body_text': body, 
                        'improvement_reasoning': 'Failed to generate improvements'
                    }
                
        except Exception as e:
            logger.error(f"Error generating copy: {e}")
            return {
                'headline_text': headline,
                'body_text': body,
                'improvement_reasoning': f'Error: {str(e)}'
            }
    
    def _parse_text_response(self, content: str, original_headline: str, original_body: str) -> Dict[str, str]:
        """
        Fallback parser for when LLM returns plain text instead of JSON.
        
        Args:
            content: Raw text response from LLM
            original_headline: Original headline to fall back to
            original_body: Original body text to fall back to
            
        Returns:
            Dictionary with parsed headline, body, and reasoning
        """
        try:
            lines = content.strip().split('\n')
            headline = original_headline
            body = original_body
            reasoning = "Generated from text response"
            
            # Try to extract structured information from text
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                
                # Look for headline indicators
                if any(keyword in line_lower for keyword in ['headline:', 'title:', 'improved headline:']):
                    if ':' in line:
                        headline = line.split(':', 1)[1].strip().strip('"\'')
                
                # Look for body text indicators
                elif any(keyword in line_lower for keyword in ['body:', 'body text:', 'improved body:']):
                    if ':' in line:
                        body_text = line.split(':', 1)[1].strip().strip('"\'')
                        if body_text:  # Only update if we got something
                            body = body_text
                
                # Look for reasoning
                elif any(keyword in line_lower for keyword in ['reason:', 'improvement:', 'changes:']):
                    if ':' in line:
                        reasoning = line.split(':', 1)[1].strip().strip('"\'')
            
            return {
                'headline_text': headline,
                'body_text': body,
                'improvement_reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"Error parsing text response: {e}")
            return {
                'headline_text': original_headline,
                'body_text': original_body,
                'improvement_reasoning': 'Could not parse response'
            }
    
    def generate_batch_improvements(self, df: pd.DataFrame, 
                                  feature_importance: Optional[pd.DataFrame] = None,
                                  batch_size: int = 5) -> pd.DataFrame:
        """
        Generate improved copy for a batch of ads from ML feature dataset.
        
        Args:
            df: DataFrame with ad feature data (from ml_ready_ad_features.csv)
            feature_importance: Feature importance from trained model
            batch_size: Number of ads to process (set low to avoid rate limits)
            
        Returns:
            DataFrame with original and improved ad copies
        """
        results = []
        
        # Process subset if dataframe is large
        if len(df) > batch_size:
            df_subset = df.head(batch_size)
            logger.info(f"Processing first {batch_size} ads from {len(df)} total")
        else:
            df_subset = df
            
        for idx, row in df_subset.iterrows():
            logger.info(f"Processing ad {idx + 1}/{len(df_subset)}")
            
            # Extract original text (these might be in different columns)
            headline = getattr(row, 'headline_text', '') or ''
            body = getattr(row, 'body_text', '') or ''
            
            if not headline and not body:
                logger.warning(f"No text found for ad {idx}, skipping")
                continue
            
            # Generate improvements
            improved = self.generate_improved_copy(
                headline=headline,
                body=body,
                ad_data=row,
                feature_importance=feature_importance
            )
            
            # Compile results
            result = {
                'ad_id': getattr(row, 'ad_id', f'ad_{idx}'),
                'original_headline': headline,
                'original_body': body,
                'improved_headline': improved['headline_text'],
                'improved_body': improved['body_text'],
                'improvement_reasoning': improved['improvement_reasoning'],
                'category': getattr(row, 'category', 'Unknown'),
                'platform': getattr(row, 'platform', 'Unknown'),
                'current_ctr': getattr(row, 'ctr', 0.0)
            }
            results.append(result)
            
            # Rate limiting - small delay between requests
            time.sleep(1)
        
        return pd.DataFrame(results)
    
    def create_evaluation_ready_dataset(self, improved_ads_df: pd.DataFrame, 
                                      original_data: Optional[pd.DataFrame] = None,
                                      dummy_metrics: bool = True) -> pd.DataFrame:
        """
        Convert improved ad copies to model-evaluation ready format with engineered features.
        
        Args:
            improved_ads_df: DataFrame with improved ad copies (from generate_batch_improvements)
            original_data: Optional original data to preserve additional metadata
            dummy_metrics: Whether to create dummy performance metrics for new ads
            
        Returns:
            DataFrame ready for model evaluation with engineered features
        """
        if not FEATURES_AVAILABLE:
            raise ImportError("Feature engineering module not available. Cannot create evaluation-ready dataset.")
        
        logger.info("Creating evaluation-ready dataset with engineered features...")
        
        # Create separate datasets for original and improved versions
        original_rows = []
        improved_rows = []
        
        for _, row in improved_ads_df.iterrows():
            base_data = {
                'ad_id': row['ad_id'],
                'campaign_id': f"campaign_{row['ad_id'].split('_')[-1]}",  # Extract from ad_id
                'headline_text': row['original_headline'],
                'body_text': row['original_body'],
                'call_to_action': '',  # Default empty CTA
                'platform': row.get('platform', 'facebook'),
                'placement': '',  # Default empty
                'ad_format': 'image',  # Default format
                'category': row.get('category', 'ecommerce')
            }
            
            # Add dummy performance metrics if requested
            if dummy_metrics:
                # Use realistic dummy values for new ads with some variation
                import random
                base_impressions = random.randint(800, 1200)  # Vary impressions
                base_ctr = row.get('current_ctr', 0.02)
                base_clicks = int(base_impressions * base_ctr)
                base_conversions = max(1, int(base_clicks * random.uniform(0.03, 0.07)))  # Vary CVR
                base_spend = base_clicks * random.uniform(1.5, 3.0)  # Vary CPC
                
                base_data.update({
                    'impressions': base_impressions,
                    'clicks': base_clicks,
                    'conversions': base_conversions,
                    'spend': base_spend,
                    'ctr': base_ctr,
                    'cvr': base_conversions / base_clicks if base_clicks > 0 else 0.05,
                    'cpc': base_spend / base_clicks if base_clicks > 0 else 2.5,
                    'cpa': base_spend / base_conversions if base_conversions > 0 else 10.0,
                    'high_performer': 0
                })
            
            # Original version
            original_row = base_data.copy()
            original_row['ad_id'] = f"{row['ad_id']}_original"
            original_rows.append(original_row)
            
            # Improved version
            improved_row = base_data.copy()
            improved_row['ad_id'] = f"{row['ad_id']}_improved"
            improved_row['headline_text'] = row['improved_headline']
            improved_row['body_text'] = row['improved_body']
            # Assume improved version will have slightly better metrics with variation
            if dummy_metrics:
                improvement_factor = random.uniform(1.1, 1.3)  # 10-30% improvement
                improved_row['clicks'] = int(improved_row['clicks'] * improvement_factor)
                improved_row['ctr'] = improved_row['ctr'] * improvement_factor
                # Slightly improve CVR too
                improved_row['conversions'] = int(improved_row['conversions'] * random.uniform(1.05, 1.25))
                improved_row['cvr'] = improved_row['conversions'] / improved_row['clicks'] if improved_row['clicks'] > 0 else improved_row['cvr']
            improved_rows.append(improved_row)
        
        # Combine all data
        all_data = original_rows + improved_rows
        comparison_df = pd.DataFrame(all_data)
        
        # Apply feature engineering
        feature_engineer = AdCopyFeatureEngineer()
        
        logger.info("Applying feature engineering pipeline...")
        comparison_df = feature_engineer.engineer_text_features(comparison_df)
        comparison_df = feature_engineer.engineer_categorical_features(comparison_df)
        comparison_df = feature_engineer.engineer_performance_features(comparison_df)
        comparison_df = feature_engineer.create_interaction_features(comparison_df)
        
        # Select final features for model compatibility
        comparison_df, selected_features = feature_engineer.select_final_features(comparison_df)
        
        logger.info(f"Created evaluation dataset with {len(comparison_df)} rows and {len(selected_features)} features")
        
        return comparison_df

def load_feature_importance(model_path: str) -> Optional[pd.DataFrame]:
    """
    Load feature importance from trained model for optimization guidance.
    
    Args:
        model_path: Path to trained model joblib file
        
    Returns:
        DataFrame with feature importance or None if loading fails
    """
    try:
        import joblib
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'feature_importance' in model_data:
            return model_data['feature_importance']
        else:
            logger.warning("No feature importance found in model file")
            return None
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Example usage functions
def generate_single_ad_improvement(headline: str, body: str, 
                                 provider: LLMProvider = LLMProvider.ANTHROPIC, model_name = None) -> Dict[str, str]:
    """
    Quick function to improve a single ad copy.
    
    Args:
        headline: Current headline
        body: Current body text
        provider: LLM provider to use
        
    Returns:
        Dictionary with improved copy
    """
    generator = AdCopyGenerator(provider=provider, model_name=model_name)
    return generator.generate_improved_copy(headline, body)

def generate_from_csv(csv_path: str, model_path: Optional[str] = None,
                     provider: LLMProvider = LLMProvider.ANTHROPIC,
                     batch_size: int = 5, output_path: Optional[str] = None,
                     create_model_ready: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate improved ad copies from ML feature CSV file.
    
    Args:
        csv_path: Path to ml_ready_ad_features.csv or similar
        model_path: Optional path to trained model for feature importance
        provider: LLM provider to use
        batch_size: Number of ads to process
        output_path: Optional path to save results CSV
        create_model_ready: Whether to create model-evaluation ready dataset
        
    Returns:
        DataFrame with original and improved copies, optionally tuple with model-ready dataset
    """
    # Load data
    logger.info(f"Loading ad data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load feature importance if model provided
    feature_importance = None
    if model_path:
        feature_importance = load_feature_importance(model_path)
        if feature_importance is not None:
            logger.info(f"Loaded feature importance with {len(feature_importance)} features")
    
    # Initialize generator
    generator = AdCopyGenerator(provider=provider)
    
    # Generate improvements
    results = generator.generate_batch_improvements(
        df=df,
        feature_importance=feature_importance,
        batch_size=batch_size
    )
    
    # Create model-evaluation ready dataset if requested
    model_ready_df = None
    if create_model_ready and FEATURES_AVAILABLE:
        model_ready_df = generator.create_evaluation_ready_dataset(results, original_data=df)
        
        # Save model-ready dataset if output path provided
        if output_path:
            base_path = output_path.rsplit('.', 1)[0]  # Remove extension
            model_ready_path = f"{base_path}_model_ready.csv"
            model_ready_df.to_csv(model_ready_path, index=False)
            logger.info(f"Model-ready dataset saved to: {model_ready_path}")
    
    # Save basic results if output path provided
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
    
    if create_model_ready and model_ready_df is not None:
        return results, model_ready_df
    else:
        return results

if __name__ == "__main__":
    # Example usage
    print("Ad Copy Generator - CTR Optimization")
    print("=====================================")
    
    # Example 1: Single ad improvement
    sample_headline = "Save 20% on Running Shoes Today"
    sample_body = "Limited-time offer on our best-selling running shoes. Free shipping on all orders."
    
    try:
        improved = generate_single_ad_improvement(sample_headline, sample_body)
        print(f"\nOriginal: {sample_headline}")
        print(f"Improved: {improved['headline_text']}")
        print(f"Reasoning: {improved['improvement_reasoning']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Batch processing (commented out to avoid accidental runs)
    # results = generate_from_csv(
    #     csv_path="../data/ml_ready_ad_features.csv",
    #     model_path="../models/ctr_predictor_production.joblib", 
    #     batch_size=3,
    #     output_path="../data/improved_ad_copies.csv"
    # )