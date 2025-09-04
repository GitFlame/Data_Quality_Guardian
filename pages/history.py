import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.database.models import DatabaseManager

def main():
    st.title("Data Quality History & Trends")
    
    # Initialize database
    db = DatabaseManager(settings.DATABASE_URL)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Recent Analysis", "Quality Trends", "Folder Analysis"])
    
    with tab1:
        display_recent_analysis(db)
    
    with tab2:
        display_quality_trends(db)
    
    with tab3:
        display_folder_analysis(db)

def display_recent_analysis(db):
    st.header("Recent Data Analysis")
    
    # Get recent analysis history
    history = db.get_dataset_history(limit=10)
    
    if not history:
        st.info("No analysis history found")
        return
    
    # Create a DataFrame for display
    df = pd.DataFrame(history)
    df['upload_time'] = pd.to_datetime(df['upload_time'])
    df['upload_time'] = df['upload_time'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Display the history table
    st.dataframe(df, use_container_width=True)
    
    # Allow viewing details of specific dataset
    selected_id = st.selectbox(
        "Select a dataset to view details:",
        options=df['id'].tolist(),
        format_func=lambda x: f"Dataset {x}: {df[df['id']==x]['name'].iloc[0]}"
    )
    
    if selected_id:
        details = db.get_dataset_details(selected_id)
        if details:
            with st.expander("Dataset Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Dataset Information")
                    st.write(details['dataset'])
                
                with col2:
                    st.markdown("#### AI Insights")
                    if details['ai_insights']:
                        st.write(details['ai_insights'])
                
                st.markdown("#### Quality Checks")
                if details['quality_checks']:
                    checks_df = pd.DataFrame(details['quality_checks'])
                    st.dataframe(checks_df, use_container_width=True)

def display_quality_trends(db):
    st.header("Quality Trends")
    
    # Time range selection
    days = st.slider("Select time range (days):", 7, 90, 30)
    
    # Get trend data
    trends = db.get_quality_trends(days)
    
    if not trends['confidence_trend']:
        st.info("No trend data available")
        return
    
    # Plot confidence trend
    conf_df = pd.DataFrame(trends['confidence_trend'])
    conf_df['date'] = pd.to_datetime(conf_df['date'])
    
    fig = px.line(
        conf_df,
        x='date',
        y='confidence',
        title='AI Confidence Score Trend',
        labels={'date': 'Date', 'confidence': 'Confidence Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot severity distribution
    if trends['severity_distribution']:
        sev_df = pd.DataFrame(trends['severity_distribution'])
        fig = px.pie(
            sev_df,
            values='count',
            names='severity',
            title='Quality Check Severity Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_folder_analysis(db):
    st.header("Folder Analysis")
    
    # Get recent folder analyses
    folders = [d for d in db.get_dataset_history(limit=50) if d['is_folder']]
    
    if not folders:
        st.info("No folder analysis found")
        return
    
    # Allow selecting a folder
    selected_folder = st.selectbox(
        "Select a folder to view analysis:",
        options=[f['id'] for f in folders],
        format_func=lambda x: f"Folder {x}: {next(f['name'] for f in folders if f['id']==x)}"
    )
    
    if selected_folder:
        analysis = db.get_folder_analysis(selected_folder)
        if analysis:
            st.subheader(f"Analysis for: {analysis['folder_name']}")
            st.metric("Total Files", analysis['file_count'])
            
            # Display file analyses
            if analysis['files']:
                files_df = pd.DataFrame(analysis['files'])
                st.dataframe(files_df, use_container_width=True)
                
                # Plot confidence distribution
                fig = px.box(
                    files_df,
                    y='confidence_score',
                    title='Confidence Score Distribution Across Files'
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
