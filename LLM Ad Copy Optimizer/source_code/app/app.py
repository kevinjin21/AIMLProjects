"""
LLM Ad Copy Optimization Results Dashboard

A simple Streamlit app to display the results of AI-powered ad copy optimization,
showing original vs improved ad copies with CTR predictions and key improvements.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Configure Streamlit page
st.set_page_config(
    page_title="LLM Ad Copy Optimization Results",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .improvement-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .ad-comparison {
        border: 1px solid #e9ecef;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_evaluation_data():
    """Load the evaluation results and ad copy data"""
    try:
        # Try to load the improved prompt results first, fall back to original
        data_dir = Path("../../data/pipeline")
        
        # Load evaluation results
        eval_file = data_dir / "evaluation_results_improved_prompt.csv"
        if not eval_file.exists():
            eval_file = data_dir / "evaluation_results.csv"
        
        if not eval_file.exists():
            return None, None, "Evaluation results file not found"
            
        evaluation_df = pd.read_csv(eval_file)
        
        # Load ad copy details
        ads_file = data_dir / "updated_prompt_improved_ads.csv"
        if not ads_file.exists():
            ads_file = data_dir / "llm_improved_ad_copies.csv"
            
        if not ads_file.exists():
            return evaluation_df, None, "Ad copy details file not found"
            
        ads_df = pd.read_csv(ads_file)
        
        # Merge the datasets on ad_id, keeping ad text data from ads_df and CTR data from evaluation_df
        # Use suffixes to handle overlapping columns
        merged_df = evaluation_df.merge(ads_df, on='ad_id', how='left', suffixes=('_eval', '_ads'))
        
        # Clean up the merged dataframe - prefer text data from ads_df
        if 'original_headline_ads' in merged_df.columns:
            merged_df['original_headline'] = merged_df['original_headline_ads']
        if 'improved_headline_ads' in merged_df.columns:
            merged_df['improved_headline'] = merged_df['improved_headline_ads']
        if 'original_body_ads' in merged_df.columns:
            merged_df['original_body'] = merged_df['original_body_ads']
        if 'improved_body_ads' in merged_df.columns:
            merged_df['improved_body'] = merged_df['improved_body_ads']
        if 'improvement_reasoning' in merged_df.columns:
            merged_df['reasoning'] = merged_df['improvement_reasoning']
        
        return evaluation_df, merged_df, None
        
    except Exception as e:
        return None, None, f"Error loading data: {str(e)}"

def format_ctr_change(value):
    """Format CTR change with appropriate color coding"""
    if value > 0:
        return f'<span class="improvement-positive">+{value:.1f}%</span>'
    elif value < 0:
        return f'<span class="improvement-negative">{value:.1f}%</span>'
    else:
        return f'<span class="improvement-neutral">{value:.1f}%</span>'

def highlight_changes(original, improved):
    """Simple highlighting of key differences between original and improved text"""
    # This is a basic implementation - could be enhanced with more sophisticated diff highlighting
    if len(improved) > len(original):
        return f"**Expanded** from {len(original)} to {len(improved)} characters"
    elif len(improved) < len(original):
        return f"**Condensed** from {len(original)} to {len(improved)} characters"
    else:
        return f"**Refined** while maintaining {len(original)} characters"

def display_ad_comparison(row):
    """Display a single ad comparison in a structured format"""
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**📋 Original Ad**")
        # Get original headline and body
        orig_headline = row.get('original_headline', 'Not available')
        orig_body = row.get('original_body', 'Not available')
        
        if pd.notna(orig_headline) and orig_headline != 'Not available':
            st.markdown(f"*Headline:* {orig_headline}")
            headline_len = len(str(orig_headline))
        else:
            st.markdown("*Headline:* Not available")
            headline_len = 0
            
        if pd.notna(orig_body) and orig_body != 'Not available':
            # Truncate body text for display
            body_display = str(orig_body)[:100]
            if len(str(orig_body)) > 100:
                body_display += "..."
            st.markdown(f"*Body:* {body_display}")
        else:
            st.markdown("*Body:* Not available")
            
        st.markdown(f"*Length:* {headline_len} chars")
    
    with col2:
        st.markdown("**🚀 Improved Ad**")
        # Get improved headline and body
        improved_headline = row.get('improved_headline', 'Not available')
        improved_body = row.get('improved_body', 'Not available')
        
        if pd.notna(improved_headline) and improved_headline != 'Not available':
            st.markdown(f"*Headline:* {improved_headline}")
            improved_len = len(str(improved_headline))
        else:
            st.markdown("*Headline:* Not available")
            improved_len = 0
            
        if pd.notna(improved_body) and improved_body != 'Not available':
            # Truncate body text for display
            body_display = str(improved_body)[:100]
            if len(str(improved_body)) > 100:
                body_display += "..."
            st.markdown(f"*Body:* {body_display}")
        else:
            st.markdown("*Body:* Not available")
            
        st.markdown(f"*Length:* {improved_len} chars")
        
        # Show length change
        if headline_len > 0 and improved_len > 0:
            length_change = improved_len - headline_len
            change_text = f"({length_change:+d} chars)" if length_change != 0 else "(same length)"
            st.markdown(f"*Change:* {change_text}")
    
    with col3:
        st.markdown("**📊 Performance**")
        ctr_change = row['ctr_improvement_pct']
        st.markdown(f"CTR Change: {format_ctr_change(ctr_change)}", unsafe_allow_html=True)
        
        # Show original and improved CTR values
        if 'predicted_ctr_original' in row and 'predicted_ctr_improved' in row:
            orig_ctr = row['predicted_ctr_original']
            improved_ctr = row['predicted_ctr_improved']
            st.markdown(f"*Original CTR:* {orig_ctr:.3f}")
            st.markdown(f"*Improved CTR:* {improved_ctr:.3f}")
        
        # Performance indicator
        if ctr_change > 0.5:
            st.success("Strong Improvement")
        elif ctr_change > 0:
            st.info("Modest Improvement")
        elif ctr_change >= -0.5:
            st.warning("Slight Decline")
        else:
            st.error("Significant Decline")

def main():
    st.title("📈 LLM Ad Copy Optimization Results")
    st.markdown("---")
    
    # Load data
    evaluation_df, merged_df, error = load_evaluation_data()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please ensure the evaluation results CSV files are in the correct directory.")
        return
    
    if evaluation_df is None:
        st.error("❌ Could not load evaluation data")
        return
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Performance filter
    performance_filter = st.sidebar.selectbox(
        "Show ads by performance:",
        ["All", "Improvements Only", "Declines Only", "Strong Improvements (≥0.5%)", "Strong Declines (≤-0.5%)"]
    )
    
    # Apply filters
    filtered_df = evaluation_df.copy()
    
    if performance_filter == "Improvements Only":
        filtered_df = filtered_df[filtered_df['ctr_improvement_pct'] > 0]
    elif performance_filter == "Declines Only":
        filtered_df = filtered_df[filtered_df['ctr_improvement_pct'] <= 0]
    elif performance_filter == "Strong Improvements (≥0.5%)":
        filtered_df = filtered_df[filtered_df['ctr_improvement_pct'] >= 0.5]
    elif performance_filter == "Strong Declines (≤-0.5%)":
        filtered_df = filtered_df[filtered_df['ctr_improvement_pct'] <= -0.5]
    
    # Overall metrics
    st.header("📊 Overall Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_ads = len(evaluation_df)
        st.metric("Total Ads Tested", total_ads)
    
    with col2:
        avg_improvement = evaluation_df['ctr_improvement_pct'].mean()
        st.metric("Average CTR Change", f"{avg_improvement:.2f}%")
    
    with col3:
        success_rate = (evaluation_df['ctr_improvement_pct'] > 0).sum() / len(evaluation_df) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        best_improvement = evaluation_df['ctr_improvement_pct'].max()
        st.metric("Best Improvement", f"+{best_improvement:.1f}%")
    
    # Performance distribution chart
    st.header("📈 Performance Distribution")
    
    # Create bins for the histogram
    bins = [-3, -1, -0.5, 0, 0.5, 1, 3]
    labels = ['Strong Decline\n(≤-1%)', 'Moderate Decline\n(-1% to -0.5%)', 'Slight Decline\n(-0.5% to 0%)', 
              'Slight Improvement\n(0% to 0.5%)', 'Moderate Improvement\n(0.5% to 1%)', 'Strong Improvement\n(≥1%)']
    
    evaluation_df['performance_category'] = pd.cut(evaluation_df['ctr_improvement_pct'], bins=bins, labels=labels, include_lowest=True)
    category_counts = evaluation_df['performance_category'].value_counts()
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title="Distribution of CTR Improvements",
        labels={'x': 'Performance Category', 'y': 'Number of Ads'},
        color=category_counts.values,
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.header("📋 Detailed Ad Comparisons")
    st.markdown(f"Showing {len(filtered_df)} of {len(evaluation_df)} ads")
    
    # Sort by performance
    sort_option = st.selectbox("Sort by:", ["Best Improvements First", "Worst Declines First", "Ad ID"])
    
    if sort_option == "Best Improvements First":
        display_df = filtered_df.sort_values('ctr_improvement_pct', ascending=False)
    elif sort_option == "Worst Declines First":
        display_df = filtered_df.sort_values('ctr_improvement_pct', ascending=True)
    else:
        display_df = filtered_df.sort_values('ad_id')
    
    # Display ads with detailed comparisons if merged data is available
    if merged_df is not None:
        try:
            for idx, (_, row) in enumerate(display_df.head(20).iterrows()):  # Limit to 20 for performance
                with st.expander(f"Ad {idx+1}: {row['ad_id']} ({row['ctr_improvement_pct']:+.1f}%)", 
                               expanded=(idx < 5)):  # Auto-expand first 5
                    
                    # Get merged data for this ad
                    merged_row = merged_df[merged_df['ad_id'] == row['ad_id']]
                    if not merged_row.empty:
                        merged_row = merged_row.iloc[0]
                        display_ad_comparison(merged_row)
                        
                        # Show AI reasoning if available
                        reasoning = merged_row.get('reasoning', merged_row.get('improvement_reasoning'))
                        if pd.notna(reasoning):
                            st.markdown("**🤖 AI Improvement Strategy:**")
                            st.markdown(f"_{reasoning}_")
                    else:
                        st.warning("Detailed ad copy data not available for this ad.")
        except Exception as e:
            st.error(f"Error displaying ad comparisons: {str(e)}")
            st.info("Falling back to basic table view...")
            # Fallback to basic table view
            basic_cols = ['ad_id', 'ctr_improvement_pct']
            available_cols = [col for col in basic_cols if col in display_df.columns]
            st.dataframe(display_df[available_cols], use_container_width=True)
    else:
        # Fallback to basic table view
        basic_cols = ['ad_id', 'ctr_improvement_pct']
        available_cols = [col for col in basic_cols if col in display_df.columns]
        st.dataframe(display_df[available_cols], use_container_width=True)
    
    # Summary insights
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ What Worked")
        successful_ads = evaluation_df[evaluation_df['ctr_improvement_pct'] > 0]
        if len(successful_ads) > 0:
            st.write(f"• {len(successful_ads)} ads showed improvement")
            st.write(f"• Average improvement: +{successful_ads['ctr_improvement_pct'].mean():.2f}%")
            st.write(f"• Best performer: +{successful_ads['ctr_improvement_pct'].max():.1f}%")
        else:
            st.write("• No ads showed improvement in this dataset")
    
    with col2:
        st.subheader("❌ What Didn't Work") 
        failed_ads = evaluation_df[evaluation_df['ctr_improvement_pct'] <= 0]
        if len(failed_ads) > 0:
            st.write(f"• {len(failed_ads)} ads showed decline or no change")
            st.write(f"• Average decline: {failed_ads['ctr_improvement_pct'].mean():.2f}%")
            st.write(f"• Worst performer: {failed_ads['ctr_improvement_pct'].min():.1f}%")
        else:
            st.write("• All ads showed improvement!")

if __name__ == "__main__":
    main()