import streamlit as st
import sys
import os
import io
import base64
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple

# Add project root to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from src.config.settings import settings
    from src.ingestion.file_handler import FileHandler
    from src.ingestion.folder_handler import FolderHandler
    from src.analysis.quality_checker import DataQualityChecker
    from src.database.models import DatabaseManager
    from src.llm_engine.llm_client import LLMClient
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Full traceback:")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

# Initialize database and LLM client
@st.cache_resource
def initialize_components(llm_provider):
    """Initialize database and LLM client"""
    db = DatabaseManager(settings.DATABASE_URL)
    llm_client = LLMClient(provider=llm_provider.lower())
    return db, llm_client

def process_single_file(uploaded_file, db, llm_client) -> Dict[str, Any]:
    """Process a single uploaded file"""
    try:
        # Parse file
        df, metadata = FileHandler.parse_file(uploaded_file)
        
        # Save to database
        dataset_id = db.save_dataset(
            name=metadata['filename'],
            file_path=f"uploads/{metadata['filename']}",
            file_size=metadata['file_size'],
            row_count=metadata['row_count'],
            column_count=metadata['column_count'],
            file_type=metadata['file_type'],
            is_folder=False
        )
        
        # Run quality checks
        quality_checker = DataQualityChecker(df)
        quality_results = quality_checker.check_data_quality()
        
        # Get AI insights
        ai_insights = llm_client.analyze_data_quality(df, quality_results, metadata)
        # Include the dataframe in ai_insights for payment status analysis
        if ai_insights:
            ai_insights['dataframe'] = df
        
        # Save AI insights to database
        if ai_insights:
            # Prepare AI insights for database storage
            db.save_ai_insight(
                dataset_id=dataset_id,
                issue_summary=ai_insights.get('summary', ''),
                business_impact=ai_insights.get('business_impact', {}),  # Can be dict or string
                suggested_fixes=ai_insights.get('suggestions', {}),
                confidence_score=ai_insights.get('confidence', 0.8)
            )
        
        return {
            'dataset_id': dataset_id,
            'dataframe': df,
            'metadata': metadata,
            'quality_results': quality_results,
            'ai_insights': ai_insights,
            'processing_status': 'success'
        }
        
    except Exception as e:
        return {
            'processing_status': 'error',
            'error_message': str(e)
        }

def process_folder(uploaded_zip, db, llm_client) -> Dict[str, Any]:
    """Process uploaded zip folder"""
    try:
        folder_handler = FolderHandler()
        
        # Extract and process all files
        processed_files, folder_metadata = folder_handler.extract_and_process_zip(uploaded_zip)
        
        # If there's only one file, process it as a single file
        if len(processed_files) == 1:
            file_data = processed_files[0]
            df = file_data['dataframe']
            metadata = file_data['metadata']
            
            # Run quality checks
            quality_checker = DataQualityChecker(df)
            quality_results = quality_checker.check_data_quality()
            
            # Get AI insights
            ai_insights = llm_client.analyze_data_quality(df, quality_results, metadata)
            if ai_insights:
                ai_insights['dataframe'] = df
            
            return {
                'dataset_id': None,  # No need to save to DB for preview
                'dataframe': df,
                'metadata': metadata,
                'quality_results': quality_results,
                'ai_insights': ai_insights,
                'processing_status': 'success'
            }
        
        # For multiple files, continue with folder processing
        # Save folder as parent dataset
        parent_dataset_id = db.save_dataset(
            name=folder_metadata['folder_name'],
            file_path=f"uploads/{folder_metadata['folder_name']}",
            file_size=folder_metadata['folder_size'],
            row_count=0,  # Will be sum of all files
            column_count=0,  # Will be sum of all files
            file_type='.zip',
            is_folder=True
        )
        
        folder_results = []
        
        # Process each file in the folder
        for file_data in processed_files:
            df = file_data['dataframe']
            metadata = file_data['metadata']
            
            # Save individual file to database
            dataset_id = db.save_dataset(
                name=metadata['filename'],
                file_path=f"uploads/{folder_metadata['folder_name']}/{file_data['file_path']}",
                file_size=metadata['file_size'],
                row_count=metadata['row_count'],
                column_count=metadata['column_count'],
                file_type=metadata['file_type'],
                is_folder=False,
                parent_dataset_id=parent_dataset_id
            )
            
            # Run quality checks
            quality_checker = DataQualityChecker(df)
            quality_results = quality_checker.check_data_quality()
            
            # Get AI insights
            ai_insights = llm_client.analyze_data_quality(df, quality_results, metadata)
            # Include the dataframe in ai_insights for payment status analysis
            if ai_insights:
                ai_insights['dataframe'] = df
            
            # Save AI insights
            if ai_insights:
                db.save_ai_insight(
                    dataset_id=dataset_id,
                    issue_summary=ai_insights.get('summary', ''),
                    business_impact=ai_insights.get('business_impact', ''),
                    suggested_fixes=ai_insights.get('suggestions', {}),
                    confidence_score=ai_insights.get('confidence', 0.8)
                )
            
            folder_results.append({
                'dataset_id': dataset_id,
                'dataframe': df,
                'metadata': metadata,
                'quality_results': quality_results,
                'ai_insights': ai_insights,
                'file_path': file_data['file_path']
            })
        
        # Get folder summary
        folder_summary = folder_handler.get_folder_summary(processed_files)
        
        return {
            'parent_dataset_id': parent_dataset_id,
            'folder_metadata': folder_metadata,
            'folder_summary': folder_summary,
            'file_results': folder_results,
            'processing_status': 'success'
        }
        
    except Exception as e:
        return {
            'processing_status': 'error',
            'error_message': str(e)
        }

def create_quality_dashboard(results: Dict[str, Any], is_folder: bool = False):
    """Create interactive quality dashboard"""
    
    if is_folder:
        folder_metadata = results['folder_metadata']
        folder_summary = results['folder_summary']
        
        # For single-file zip folders, skip folder-level display
        if folder_metadata['total_files'] == 1:
            # Display only the individual file analysis
            display_single_file_analysis(results['file_results'][0])
            return
            
        # Folder dashboard for multiple files
        st.subheader(" Folder Analysis Dashboard")
        
        # Folder overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" Total Files", folder_metadata['total_files'])
        with col2:
            st.metric(" Processed", folder_metadata['processed_files'])
        with col3:
            st.metric(" Failed", folder_metadata['failed_files'])
        with col4:
            st.metric(" Total Size", f"{folder_metadata['folder_size'] / 1024:.1f} KB")
        
        # File type distribution
        if folder_metadata['supported_files']:
            file_extensions = [Path(f).suffix for f in folder_metadata['supported_files']]
            ext_counts = pd.Series(file_extensions).value_counts()
            
            fig = px.pie(
                values=ext_counts.values,
                names=ext_counts.index,
                title=" File Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Per-file analysis
        st.subheader(" Individual File Analysis")
        
        for i, file_result in enumerate(results['file_results']):
            with st.expander(f" {file_result['metadata']['filename']}"):
                display_single_file_analysis(file_result)
    
    else:
        # Single file dashboard
        display_single_file_analysis(results)

def display_single_file_analysis(result: Dict[str, Any]):
    """Display analysis for a single file"""
    metadata = result['metadata']
    
    # If ai_insights exists, add file identification information
    if 'ai_insights' in result:
        if isinstance(result['ai_insights'], dict):
            result['ai_insights']['file_id'] = metadata.get('file_id')
            result['ai_insights']['filename'] = metadata.get('filename')
    quality_results = result['quality_results']
    ai_insights = result.get('ai_insights', {})
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{metadata['row_count']:,}")
    with col2:
        st.metric("📋 Columns", metadata['column_count'])
    with col3:
        st.metric("💾 Size", f"{metadata['file_size'] / 1024:.1f} KB")
    with col4:
        st.metric("📄 Type", metadata['file_type'].upper())
    
    # Quality Score Card
    overall_score = calculate_quality_score(quality_results)
    
    # Color based on score
    if overall_score >= 80:
        score_color = "🟢"
    elif overall_score >= 60:
        score_color = "🟡"
    else:
        score_color = "🔴"
    
    st.markdown(f"### {score_color} Overall Quality Score: {overall_score}/100")
    
    # Custom CSS for tabs
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 0px 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0e2e6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            border: 2px solid #4a90e2 !important;
            border-radius: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create basic tabs list
    tab_names = [" Overview", " Missing Data", " Duplicates"]
    
    # Check if this is a financial dataset by looking for payment and amount columns in the dataframe
    df = quality_results.get('dataframe', None)
    if df is not None:
        # Flexible column detection using partial matches
        payment_cols = [col for col in df.columns if any(payment_term in col.lower() for payment_term in ['payment', 'status', 'paid', 'mode'])]
        amount_cols = [col for col in df.columns if any(amount_term in col.lower() for amount_term in ['amount', 'total', 'price', 'cost'])]
        client_cols = [col for col in df.columns if any(client_term in col.lower() for client_term in ['client', 'customer', 'project', 'name'])]
        invoice_cols = [col for col in df.columns if any(invoice_term in col.lower() for invoice_term in ['invoice', 'transaction', 'id', 'reference', 'code'])]
        
        is_financial = bool(payment_cols and amount_cols and (client_cols or invoice_cols))
        
        if is_financial:
            tab_names.append(" Payment Status")
    
    # Add remaining tabs
    tab_names.extend([" AI Insights", "Data Cleaning"])
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Basic tabs
    with tabs[0]:
        display_overview_tab(quality_results)
    
    with tabs[1]:
        display_missing_data_tab(quality_results)
    
    with tabs[2]:
        display_duplicates_tab(quality_results)
    
    # Initialize current tab index
    current_tab = 3
    
    # Conditionally display payment status tab if it exists
    if any("Payment Status" in tab for tab in tab_names):
        with tabs[current_tab]:
            display_payment_status_tab(ai_insights)
        current_tab += 1
    
    # Display AI Insights tab
    with tabs[current_tab]:
        display_ai_insights_tab(ai_insights)
    
    # Display Data Cleaning tab
    with tabs[current_tab + 1]:
        display_data_cleaning_tab(quality_results)

def calculate_quality_score(quality_results: Dict[str, Any]) -> int:
    """Calculate overall quality score (0-100)"""
    score = 100
    
    # Deduct points for missing data
    missing_percentage = quality_results['missing_data']['missing_data_percentage']
    score -= min(missing_percentage * 0.5, 30)  # Max 30 points deduction
    
    # Deduct points for duplicates
    duplicates = quality_results['duplicates']
    row_duplicate_percentage = duplicates['row_duplicates']['duplicate_percentage']
    column_duplicate_count = duplicates['column_duplicates']['total_duplicate_columns']
    value_duplicate_count = len(duplicates['value_duplicates'])
    
    # Deduct for row duplicates
    score -= min(row_duplicate_percentage * 0.2, 15)  # Max 15 points deduction
    
    # Deduct for column duplicates
    score -= min(column_duplicate_count * 5, 10)  # Max 10 points deduction
    
    # Deduct for value duplicates
    score -= min(value_duplicate_count * 1, 10)  # Max 10 points deduction
    
    # Deduct points for consistency issues
    consistency_issues = quality_results['consistency']
    issue_count = sum(len(issues) for issues in consistency_issues.values())
    score -= min(issue_count * 2, 15)  # Max 15 points deduction
    
    return max(int(score), 0)

def display_overview_tab(quality_results: Dict[str, Any]):
    """Display overview tab with summary statistics"""
    summary = quality_results['summary']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Data Summary")
        st.metric("Total Cells", f"{summary['total_rows'] * summary['total_columns']:,}")
        st.metric("Numeric Columns", summary['numeric_columns'])
        st.metric("Text Columns", summary['categorical_columns'])
        st.metric("Date Columns", summary['datetime_columns'])
    
    with col2:
        st.markdown("####  Quality Metrics")
        
        # Missing data percentage
        missing_pct = quality_results['missing_data']['missing_data_percentage']
        st.metric("Missing Data %", f"{missing_pct:.1f}%")
        
        # Duplicate percentages
        duplicates = quality_results['duplicates']
        row_duplicate_pct = duplicates['row_duplicates']['duplicate_percentage']
        column_duplicate_count = duplicates['column_duplicates']['total_duplicate_columns']
        value_duplicate_count = len(duplicates['value_duplicates'])
        
        st.metric("Duplicate Rows %", f"{row_duplicate_pct:.1f}%")
        if column_duplicate_count > 0:
            st.metric("Duplicate Columns", column_duplicate_count)
        if value_duplicate_count > 0:
            st.metric("Columns with Duplicates", value_duplicate_count)
        
        # Memory usage
        memory_mb = summary['memory_usage'] / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")

def display_missing_data_tab(quality_results: Dict[str, Any]):
    """Display missing data analysis"""
    missing_data = quality_results['missing_data']
    
    # Overall missing data stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Missing Cells", f"{missing_data['total_missing_cells']:,}")
    with col2:
        st.metric("Overall Missing %", f"{missing_data['missing_data_percentage']:.2f}%")
    
    # Per-column missing data
    if missing_data['total_missing_cells'] > 0:
        st.markdown("####  Missing Data by Column")
        
        # Create dataframe for visualization
        missing_df = pd.DataFrame(missing_data['by_column']).T
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
        
        if not missing_df.empty:
            # Bar chart of missing percentages
            fig = px.bar(
                x=missing_df.index,
                y=missing_df['missing_percentage'],
                title="Missing Data Percentage by Column",
                labels={'x': 'Column', 'y': 'Missing %'},
                color=missing_df['missing_percentage'],
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing data found!")
    else:
        st.success("No missing data found!")

def clean_duplicate_data(df: pd.DataFrame, cleaning_options: List[str], 
                      quality_results: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Clean data based on selected options and return cleaned dataframe and summary"""
    cleaned_df = df.copy()
    cleaning_summary = []
    
    try:
        duplicates = quality_results['duplicates']
        row_duplicates = duplicates['row_duplicates']
        column_duplicates = duplicates['column_duplicates']
        missing_data = quality_results['missing_data']
        
        if "Remove duplicate rows" in cleaning_options:
            # Get counts before cleaning
            original_rows = len(cleaned_df)
            expected_removals = row_duplicates['removable_duplicates']
            total_in_duplicate_sets = row_duplicates['total_duplicates']
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(keep='first')
            actual_removed = original_rows - len(cleaned_df)
            
            # Add to cleaning summary
            if actual_removed > 0:
                if actual_removed == expected_removals:
                    cleaning_summary.append(
                        f"- Removed {actual_removed:,} duplicate rows "
                        f"(from {total_in_duplicate_sets:,} total rows in duplicate sets)"
                    )
                else:
                    # This should never happen, but we'll handle it just in case
                    cleaning_summary.append(
                        f"- Warning: Removed {actual_removed:,} rows, but expected to remove {expected_removals:,}"
                    )
        
        if "Remove duplicate columns" in cleaning_options and column_duplicates['total_duplicate_columns'] > 0:
            duplicate_cols = set()
            for original_col, duplicates_info in column_duplicates['duplicate_columns'].items():
                for dup_info in duplicates_info:
                    duplicate_cols.add(dup_info['column'])
            if duplicate_cols:
                cleaned_df = cleaned_df.drop(columns=list(duplicate_cols))
                cleaning_summary.append(f"- Removed {len(duplicate_cols)} duplicate columns")
        
        if "Fix inconsistent values" in cleaning_options:
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
            changes_made = False
            for col in text_columns:
                if cleaned_df[col].notna().any():
                    original_values = cleaned_df[col].copy()
                    cleaned_df[col] = cleaned_df[col].str.strip().str.lower()
                    if not original_values.equals(cleaned_df[col]):
                        changes_made = True
            if changes_made:
                cleaning_summary.append("- Standardized text values in all text columns")
        
        if "Drop columns with high missing values" in cleaning_options:
            threshold = 0.5  # 50% missing values threshold
            high_missing_cols = [
                col for col, info in missing_data['by_column'].items()
                if info['missing_percentage'] > threshold * 100
            ]
            if high_missing_cols:
                cleaned_df = cleaned_df.drop(columns=high_missing_cols)
                cleaning_summary.append(f"- Removed {len(high_missing_cols)} columns with >50% missing values")
        
        if "Fill missing values" in cleaning_options:
            numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
            
            changes_made = False
            # Fill numeric columns with median
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    changes_made = True
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                if cleaned_df[col].isnull().any():
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    changes_made = True
            
            if changes_made:
                cleaning_summary.append("- Filled missing values (median for numeric, mode for categorical)")
        
        if not cleaning_summary:
            cleaning_summary.append("No changes were necessary for the selected options")
            
    except Exception as e:
        cleaning_summary.append(f"Error during cleaning: {str(e)}")
        st.error(f"An error occurred while cleaning the data: {str(e)}")
        
    return cleaned_df, cleaning_summary

def display_data_cleaning_tab(quality_results: Dict[str, Any]):
    """Display data cleaning options and tools"""
    st.markdown("## Data Cleaning Tools")
    
    if not quality_results or 'dataframe' not in quality_results:
        st.warning("No data available for cleaning")
        return
        
    # Initialize session state for cleaning options and format
    if 'cleaning_options' not in st.session_state:
        st.session_state.cleaning_options = []  # Start with empty selection
    if 'download_format' not in st.session_state:
        st.session_state.download_format = "CSV"
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
        
    df = quality_results.get('dataframe')
    duplicates = quality_results['duplicates']
    row_duplicates = duplicates['row_duplicates']
    column_duplicates = duplicates['column_duplicates']
    
    # Show data quality issues summary
    st.markdown("### Current Data Quality Issues")
    
    has_duplicates = row_duplicates['total_duplicates'] > 0 or column_duplicates['total_duplicate_columns'] > 0
    
    if not has_duplicates:
        st.success("✅ No duplicate rows or columns found in the dataset. No cleaning needed!")
        return

    issues_col1, issues_col2 = st.columns(2)
    with issues_col1:
        if row_duplicates['removable_duplicates'] > 0:
            total_dupes = row_duplicates['total_duplicates']
            removable = row_duplicates['removable_duplicates']
            st.warning(
                f"🔄 Found {total_dupes:,} rows in duplicate sets "
                f"({removable:,} can be removed)"
            )
        if column_duplicates['total_duplicate_columns'] > 0:
            st.warning(f"📊 {column_duplicates['total_duplicate_columns']} duplicate columns found")
            
    with issues_col2:
        missing_data = quality_results['missing_data']
        if missing_data['total_missing_cells'] > 0:
            st.warning(f"❓ {missing_data['total_missing_cells']:,} missing values found")
    
    # Cleaning Options
    st.markdown("### 🧹 Data Cleaning Analysis")
    
    # Show only duplicate cleaning options
    available_options = [
        "Remove duplicate rows",
        "Remove duplicate columns"
    ]
    
    def on_cleaning_options_change():
        st.session_state.cleaned_df = None  # Reset cleaned data when options change
        
    def on_format_change():
        st.session_state.download_format = st.session_state.format_select
    
    cleaning_options = st.multiselect(
        "Select cleaning options:",
        available_options,
        default=[],  # Start with no default selections
        help="Choose cleaning actions to apply",
        key="cleaning_select",
        on_change=on_cleaning_options_change
    )
    # Update session state
    st.session_state.cleaning_options = cleaning_options

    st.selectbox(
        "Select download format:",
        ("CSV", "Excel"),
        key="format_select",
        index=0 if st.session_state.download_format == "CSV" else 1,
        on_change=on_format_change
    )

    if st.button("🧹 Clean Data", key="clean_data_btn"):
        try:
            # If no cleaning options selected, use original data
            if cleaning_options:
                cleaned_df, cleaning_summary = clean_duplicate_data(
                    quality_results['dataframe'], cleaning_options, quality_results
                )
                st.session_state.cleaned_df = cleaned_df
                st.success("✅ Data cleaned successfully!")
            else:
                st.session_state.cleaned_df = quality_results['dataframe']
                cleaning_summary = ["No cleaning options selected. Using original data."]
                st.info("No cleaning performed. Using original data.")

            # Show cleaning summary directly
            st.markdown("### 🧹 Cleaning Summary")
            for item in cleaning_summary:
                st.markdown(item)

            # Show sample of cleaned data
            st.markdown("### 👀 Preview of Cleaned Data")
            st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error during cleaning: {str(e)}")
            return

    # Only show download options if we have cleaned data
    if st.session_state.cleaned_df is not None:
        st.markdown("### 📥 Download Cleaned Data")
        
        try:
            if st.session_state.download_format == "CSV":
                csv_data = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    key="download_cleaned_csv"
                )
            else:  # Excel
                excel_buffer = io.BytesIO()
                st.session_state.cleaned_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_cleaned_excel"
                )
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")

            # Show cleaning statistics
            st.markdown("### 📊 Cleaning Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                original_df = quality_results['dataframe']
                rows_removed = len(original_df) - len(cleaned_df)
                st.metric(
                    "Rows Cleaned", 
                    f"{rows_removed:,}",
                    help="Number of rows removed during cleaning"
                )

            with stats_col2:
                cols_removed = len(original_df.columns) - len(cleaned_df.columns)
                st.metric(
                    "Columns Cleaned",
                    f"{cols_removed:,}",
                    help="Number of columns removed during cleaning"
                )

            with stats_col3:
                total_cells_cleaned = (len(original_df) * len(original_df.columns)) - (len(cleaned_df) * len(cleaned_df.columns))
                st.metric(
                    "Total Cells Affected",
                    f"{total_cells_cleaned:,}",
                    help="Total number of data cells affected by cleaning"
                )
        except Exception as e:
            st.error(f"Error during cleaning: {str(e)}")
            st.exception(e)

def display_duplicates_tab(quality_results: Dict[str, Any]):
    """Display comprehensive duplicate analysis"""
    duplicates = quality_results['duplicates']
    row_duplicates = duplicates['row_duplicates']
    column_duplicates = duplicates['column_duplicates']
    value_duplicates = duplicates['value_duplicates']
    
    st.markdown("### Row-Level Duplicates")
    # Row duplicate metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duplicate Rows", f"{row_duplicates['total_duplicates']:,}")
    with col2:
        st.metric("Duplicate %", f"{row_duplicates['duplicate_percentage']:.2f}%")
    with col3:
        st.metric("Unique Rows", f"{row_duplicates['unique_rows']:,}")
    
    if row_duplicates['total_duplicates'] > 0:
        st.info("💡 You can remove duplicates in the 🧹 Data Cleaning tab")
    
    # Create pie chart for row distribution
    if row_duplicates['total_duplicates'] > 0:
        pie_data = pd.DataFrame([
            {'Category': 'Unique Rows', 'Count': row_duplicates['unique_rows']},
            {'Category': 'Duplicate Rows', 'Count': row_duplicates['total_duplicates']}
        ])
        
        fig = px.pie(
            pie_data,
            values='Count',
            names='Category',
            title='Distribution of Unique vs Duplicate Rows',
            color='Category',
            color_discrete_map={
                'Unique Rows': '#00CC96',
                'Duplicate Rows': '#EF553B'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show duplicate row examples if any
    if 'duplicate_examples' in row_duplicates and row_duplicates['duplicate_examples']:
        st.markdown("#### Examples of Duplicate Rows:")
        for idx, example in enumerate(row_duplicates['duplicate_examples'], 1):
            st.markdown(f"**Duplicate Set {idx}** (Found {example['occurrence_count']} times)")
            # Convert to dataframe for better display
            example_df = pd.DataFrame([example['row_content']])
            st.dataframe(example_df, use_container_width=True)
    
    # Column duplicates
    st.markdown("### Column-Level Duplicates")
    
    # Prepare data for visualization
    has_exact_dupes = column_duplicates['total_duplicate_columns'] > 0
    has_similar = column_duplicates.get('similar_columns', {}) and any(column_duplicates['similar_columns'].values())
    
    if has_exact_dupes or has_similar:
        # Create network graph data
        nodes = []
        edges = []
        
        # Add exact duplicates
        if has_exact_dupes:
            st.warning(f"Found {column_duplicates['total_duplicate_columns']} exact duplicate columns")
            for original_col, duplicates_info in column_duplicates['duplicate_columns'].items():
                nodes.append(original_col)
                duplicate_cols = [d['column'] for d in duplicates_info]
                for dup_col in duplicate_cols:
                    nodes.append(dup_col)
                    edges.append({
                        'from': original_col,
                        'to': dup_col,
                        'similarity': 100,
                        'type': 'exact'
                    })
                st.markdown(f"- Column **{original_col}** is identical to: {', '.join(duplicate_cols)}")
        
        # Add similar columns
        if has_similar:
            st.markdown("#### Similar Columns (80% or higher similarity):")
            for original_col, similar_info in column_duplicates['similar_columns'].items():
                nodes.append(original_col)
                for info in similar_info:
                    nodes.append(info['column'])
                    edges.append({
                        'from': original_col,
                        'to': info['column'],
                        'similarity': info['similarity'],
                        'type': 'similar'
                    })
                    st.markdown(f"- Column **{original_col}** is {info['similarity']}% similar to **{info['column']}**")
        
        # Create visualization using plotly
        if edges:
            # Create a force-directed graph
            edge_trace = []
            node_trace = []
            
            # Create similarity matrix for the heatmap
            unique_cols = list(set(nodes))
            similarity_values = [
                100 if i == j else next(
                    (e['similarity'] for e in edges 
                     if (e['from'] == unique_cols[i] and e['to'] == unique_cols[j]) or 
                     (e['from'] == unique_cols[j] and e['to'] == unique_cols[i])), 
                    0
                )
                for i in range(len(unique_cols))
                for j in range(len(unique_cols))
            ]
            
            similarity_matrix = pd.DataFrame(
                np.array(similarity_values).reshape(len(unique_cols), len(unique_cols)),
                index=unique_cols,
                columns=unique_cols
            )
            
            similarity_matrix.index = unique_cols
            similarity_matrix.columns = unique_cols
            
            # Create heatmap
            fig = px.imshow(
                similarity_matrix,
                title='Column Similarity Matrix',
                labels={'x': 'Columns', 'y': 'Columns', 'color': 'Similarity %'},
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            fig.update_layout(
                width=600,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No duplicate or similar columns found")
    
    # Value duplicates within columns with visualizations
    st.markdown("### Value-Level Duplicates")
    if value_duplicates:
        # Create summary DataFrame for all columns
        summary_data = []
        for column, details in value_duplicates.items():
            summary_data.append({
                'Column': column,
                'Duplicate Values': details['duplicate_count'],
                'Affected Rows': details['total_duplicate_rows'],
                'Percentage': details['duplicate_percentage']
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Bar chart showing duplicate percentages across columns
            fig = px.bar(
                summary_df,
                x='Column',
                y='Percentage',
                title='Duplicate Values Distribution Across Columns',
                labels={'Percentage': 'Rows with Duplicates (%)'},
                color='Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis per column
            for column, details in value_duplicates.items():
                st.markdown(f"#### Column: {column}")
                
                # Column-level metrics
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Duplicate Values", details['duplicate_count'])
                with mcol2:
                    st.metric("Affected Rows", details['total_duplicate_rows'])
                with mcol3:
                    st.metric("% of Rows", f"{details['duplicate_percentage']:.1f}%")
                
                # Create DataFrame for value distribution
                dupes_data = []
                for value, value_info in details['duplicate_values'].items():
                    dupes_data.append({
                        'Value': value,
                        'Occurrences': value_info['count'],
                        'Percentage': value_info['percentage']
                    })
                dupes_df = pd.DataFrame(dupes_data)
                
                # Create two columns for table and chart
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Table view
                    st.dataframe(dupes_df.style.format({
                        'Percentage': '{:.1f}%'
                    }), use_container_width=True)
                
                with col2:
                    if not dupes_df.empty:
                        # Pie chart for value distribution
                        fig = px.pie(
                            dupes_df,
                            values='Occurrences',
                            names='Value',
                            title=f'Distribution of Duplicate Values in {column}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
    else:
        st.success("No duplicate values found within columns")
    
    # Overall severity assessment
    st.markdown("### Overall Duplicate Assessment")
    severity = duplicates['severity']
    if severity == "High":
        st.error("🔴 **HIGH SEVERITY**: Multiple duplicate issues detected")
        st.markdown("**Recommendation**: Immediate review and deduplication required")
    elif severity == "Medium":
        st.warning("🟡 **MEDIUM SEVERITY**: Some duplicate issues detected")
        st.markdown("**Recommendation**: Review and clean identified duplicates")
    elif severity == "Low":
        st.info("🔵 **LOW SEVERITY**: Minor duplicate issues detected")
        st.markdown("**Recommendation**: Monitor and review if needed")
    elif severity == "Good":
        st.success("✅ **GOOD**: No significant duplicate issues detected")
    else:
        st.info(f"ℹ️ **DUPLICATE STATUS**: Review recommended")

def display_payment_status_tab(ai_insights: Dict[str, Any]):
    """Display payment status analysis in a dedicated tab"""
    if not ai_insights or not ai_insights.get('dataframe') is not None:
        st.warning("No data available for payment analysis")
        return
        
    df = ai_insights.get('dataframe')
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return

    try:
        # Detect financial/payment related columns
        payment_cols = [col for col in df.columns if any(term in col.lower() for term in ['payment', 'paid', 'mode', 'status'])]
        amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'total', 'price'])]
        client_cols = [col for col in df.columns if any(term in col.lower() for term in ['project', 'client', 'customer', 'name'])]
        email_cols = [col for col in df.columns if any(term in col.lower() for term in ['email', 'mail'])]
        invoice_cols = [col for col in df.columns if any(term in col.lower() for term in ['id', 'code', 'reference', 'invoice'])]
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time'])]

        # Check if this is a financial dataset - more flexible detection
        if not (payment_cols or amount_cols):
            st.info("This doesn't appear to be a financial dataset with payment information.")
            return

        # Get column names (use first found)
        payment_col = payment_cols[0] if payment_cols else None
        client_col = client_cols[0] if client_cols else None
        email_col = email_cols[0] if email_cols else None
        amount_col = amount_cols[0] if amount_cols else None
        invoice_col = invoice_cols[0] if invoice_cols else None
        date_col = date_cols[0] if date_cols else None

        if not payment_col:
            st.warning("No payment status column detected in the dataset.")
            return

        # Process payment data
        payment_data = []
        for _, row in df.iterrows():
            try:
                status = str(row[payment_col]).strip().lower()
                
                # Classify payment status
                if status in ['yes', 'paid', '1', 'true', 'done', 'complete', 'completed']:
                    payment_status = "Paid"
                elif status in ['partial', 'partially paid', 'incomplete']:
                    payment_status = "Partially Paid"
                elif status in ['no', 'unpaid', '0', 'false', 'pending', 'due']:
                    payment_status = "Not Paid"
                else:
                    payment_status = "❓ Unknown"

                # Build row data with error handling for each field
                row_data = {
                    'Client': str(row[client_col]) if client_col else 'N/A',
                    'Email': str(row[email_col]) if email_col else 'N/A',
                    'Invoice/ID': str(row[invoice_col]) if invoice_col else 'N/A',
                    'Amount': 'N/A',
                    'Status': payment_status,
                    'Due Date': 'N/A'
                }

                # Handle amount conversion safely
                if amount_col and pd.notna(row[amount_col]):
                    try:
                        row_data['Amount'] = f"${float(row[amount_col]):.2f}"
                    except (ValueError, TypeError):
                        pass

                # Handle date conversion safely
                if date_col and pd.notna(row[date_col]):
                    try:
                        row_data['Due Date'] = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        pass

                payment_data.append(row_data)
            except Exception as e:
                st.warning(f"Skipped processing a row due to error: {str(e)}")
                continue

        if not payment_data:
            st.warning("No valid payment data could be processed.")
            return

        try:
            # Convert to DataFrame
            payment_df = pd.DataFrame(payment_data)

            # Calculate financial metrics with error handling
            def safe_amount_sum(amounts, status_filter=None):
                try:
                    if status_filter is not None:
                        amounts = [amt for i, amt in enumerate(amounts) 
                                if amt != 'N/A' and payment_df['Status'].iloc[i] == status_filter]
                    else:
                        amounts = [amt for amt in amounts if amt != 'N/A']
                    
                    return sum(float(amt.replace('$','')) for amt in amounts)
                except Exception:
                    return 0.0

            total_amount = safe_amount_sum(payment_df['Amount'])
            paid_amount = safe_amount_sum(payment_df['Amount'], "Paid")
            unpaid_amount = safe_amount_sum(payment_df['Amount'], "Not Paid")
            partial_amount = safe_amount_sum(payment_df['Amount'], "Partially Paid")

            # Display summary metrics in two rows
            st.markdown("#### 📊 Transaction Metrics")
            col1, col2, col3, col4 = st.columns(4)
            total = len(payment_df)

            with col1:
                st.metric("Total Records", total)
                st.metric("Total Amount", f"${total_amount:,.2f}")
            
            with col2:
                try:
                    paid = len(payment_df[payment_df['Status'] == "Paid"])
                    paid_percent = paid/total*100 if total > 0 else 0
                    st.metric("Paid", f"{paid} ({paid_percent:.1f}%)")
                    st.metric("Paid Amount", f"${paid_amount:,.2f}")
                except Exception as e:
                    st.error(f"Error calculating paid metrics: {str(e)}")

            with col3:
                try:
                    unpaid = len(payment_df[payment_df['Status'] == "Not Paid"])
                    unpaid_percent = unpaid/total*100 if total > 0 else 0
                    st.metric("Unpaid", f"{unpaid} ({unpaid_percent:.1f}%)")
                    st.metric("Unpaid Amount", f"${unpaid_amount:,.2f}")
                except Exception as e:
                    st.error(f"Error calculating unpaid metrics: {str(e)}")

            with col4:
                try:
                    partial = len(payment_df[payment_df['Status'] == "Partially Paid"])
                    partial_percent = partial/total*100 if total > 0 else 0
                    st.metric("Partial", f"{partial} ({partial_percent:.1f}%)")
                    st.metric("Partial Amount", f"${partial_amount:,.2f}")
                except Exception as e:
                    st.error(f"Error calculating partial payment metrics: {str(e)}")

            # Additional metrics
            st.markdown("#### 📈 Financial Health Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    collection_rate = (paid_amount / total_amount * 100) if total_amount > 0 else 0
                    st.metric("Collection Rate", f"{collection_rate:.1f}%")
                except Exception as e:
                    st.error(f"Error calculating collection rate: {str(e)}")
            
            with col2:
                try:
                    at_risk_amount = unpaid_amount + partial_amount
                    st.metric("At Risk Amount", f"${at_risk_amount:,.2f}")
                except Exception as e:
                    st.error(f"Error calculating at risk amount: {str(e)}")

            # Group by client view if client information is available
            if client_col:
                st.markdown("#### 📊 Payment Status by Client")
                
                try:
                    # Filter out N/A clients and create tabs
                    valid_clients = [c for c in payment_df['Client'].unique() if c != 'N/A']
                    if valid_clients:
                        client_tabs = st.tabs([f"📋 {client}" for client in valid_clients])
                        
                        for tab, client in zip(client_tabs, valid_clients):
                            with tab:
                                client_data = payment_df[payment_df['Client'] == client]
                                
                                # Client details
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    email = client_data['Email'].iloc[0]
                                    if email != 'N/A':
                                        st.markdown(f"**Email:** {email}")
                                
                                with col2:
                                    client_total = safe_amount_sum(client_data['Amount'])
                                    if client_total > 0:
                                        st.markdown(f"**Total Amount:** ${client_total:,.2f}")
                                
                                # Client transactions
                                st.dataframe(
                                    client_data[['Invoice/ID', 'Amount', 'Status', 'Due Date']],
                                    use_container_width=True
                                )
                    else:
                        st.info("No client information available")
                
                except Exception as e:
                    st.error(f"Error displaying client data: {str(e)}")

            # Show full transaction table
            st.markdown("#### 📊 All Transactions")
            st.dataframe(payment_df, use_container_width=True)

            # Download option
            csv_data = payment_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Payment Status Report",
                data=csv_data,
                file_name="payment_status_report.csv",
                mime="text/csv",
                key=f"download_payment_status_{id(df)}"
            )

        except Exception as e:
            st.error(f"Error processing payment data: {str(e)}")
    except Exception as e:
        st.error(f"Error analyzing payment data: {str(e)}")

def display_ai_insights_tab(ai_insights: Dict[str, Any]):
    """Display AI-powered insights and suggestions"""
    import io
    import uuid
    from fpdf import FPDF

    if not ai_insights:
        st.info(" AI analysis is being processed...")
        return

    # Get a unique identifier for this specific AI insights instance
    file_id = ai_insights.get('file_id', '')  # Try to get file ID first
    if not file_id:
        # If no file ID, use filename or generate a random ID
        filename = ai_insights.get('filename', '')
        file_id = filename if filename else str(uuid.uuid4())
    
    # Ensure the ID is unique by adding a random component
    unique_instance_id = f"{file_id}_{str(uuid.uuid4())}"

    # Start with regular AI insights sections
    st.markdown("### 🤖 AI Analysis Results")
    
    # Issue summary
    if 'summary' in ai_insights and ai_insights['summary']:
        st.markdown("####  AI Analysis Summary")
        st.markdown(ai_insights['summary'])
    
    # Business impact
    if 'business_impact' in ai_insights and ai_insights['business_impact']:
        st.markdown("####  Business Impact Assessment")
        business_impact = ai_insights['business_impact']
        
        if isinstance(business_impact, dict):
            # Display structured business impact
            for impact_type, description in business_impact.items():
                st.markdown(f"**{impact_type.title()}**:")
                st.markdown(description)
        else:
            # Display string business impact
            st.markdown(business_impact)
    
    # Suggestions
    if 'suggestions' in ai_insights and ai_insights['suggestions']:
        st.markdown("####  AI Recommendations")
        suggestions = ai_insights['suggestions']
        
        if isinstance(suggestions, dict):
            for category, suggestion_list in suggestions.items():
                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                if isinstance(suggestion_list, list):
                    for suggestion in suggestion_list:
                        st.markdown(f"- {suggestion}")
                else:
                    st.markdown(f"- {suggestion_list}")
        else:
            st.markdown(suggestions)
    
    # Confidence score
    if 'confidence' in ai_insights:
        try:
            confidence = float(ai_insights['confidence']) * 100
            st.markdown(f"**AI Confidence Score**: {confidence:.1f}%")
        except (ValueError, TypeError):
            st.markdown(f"**AI Confidence Score**: {ai_insights['confidence']}")
    
    # PDF Generation
    def generate_pdf(ai_insights):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def flatten_content(content):
            # Converts dicts/lists to readable string for PDF
            if isinstance(content, dict):
                lines = []
                for k, v in content.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{k.title()}:")
                        sub = flatten_content(v)
                        for subline in sub.split('\n'):
                            lines.append(f"  # {subline}")
                    else:
                        lines.append(f"{k.title()}: {v}")
                return '\n'.join(lines)
            elif isinstance(content, list):
                return '\n'.join([f"- {flatten_content(item)}" for item in content])
            else:
                return str(content)

        def write_section(title, content):
            pdf.set_font("Arial", 'B', 16)
            pdf.set_text_color(40, 40, 120)
            pdf.cell(0, 12, title, ln=True, align='L')
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            text = flatten_content(content)
            pdf.multi_cell(0, 8, text, border=0)
            pdf.ln(2)

        # Title
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(30, 30, 100)
        pdf.cell(0, 16, "AI Data Quality Insights Report", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        # Write AI Insights sections to PDF using same logic as Streamlit display
        if 'summary' in ai_insights and ai_insights['summary']:
            write_section("AI Analysis Summary", ai_insights['summary'])

        if 'business_impact' in ai_insights and ai_insights['business_impact']:
            write_section("Business Impact Assessment", ai_insights['business_impact'])

        if 'suggestions' in ai_insights and ai_insights['suggestions']:
            write_section("AI Recommendations", ai_insights['suggestions'])

        if 'confidence' in ai_insights:
            try:
                confidence = float(ai_insights['confidence']) * 100
                write_section("AI Confidence Score", f"{confidence:.1f}%")
            except (ValueError, TypeError):
                write_section("AI Confidence Score", str(ai_insights['confidence']))

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)
        return pdf_buffer
    # Download PDF button
    pdf_buffer = generate_pdf(ai_insights)
    
    # Sanitize filename for the output file
    filename = ai_insights.get('filename', 'report')
    sanitized_filename = filename.replace('.', '_').replace(' ', '_')
    
    st.download_button(
        label="📄 Download AI Insights as PDF",
        data=pdf_buffer,
        file_name=f"ai_insights_{sanitized_filename}.pdf",
        mime="application/pdf",
        key=f"download_ai_insights_pdf_{unique_instance_id}"
    )

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Data Quality Guardian",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to center title
    st.markdown(
        """
        <style>
        .center-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
        }
        .center-subtitle {
            text-align: center;
            font-size: 1.2em;
            color: grey;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header with icons
    st.markdown(
        '<p class="center-title">🛡️ Welcome to <span style="color:#4CAF50;">Data Quality Guardian</span> 📊</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="center-subtitle">🤖 Your AI-powered solution for comprehensive data quality analysis and monitoring</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    
    # Initialize components
    # Sidebar for file upload
    llm_provider = "Gemini"  
    try:
        db, llm_client = initialize_components(llm_provider)
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.error("Make sure your .env file is configured with API keys")
        st.stop()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Data")
        
        # File type selection
        upload_type = st.radio(
            "Choose upload type:",
            ["Single File", "Folder (Zip)"],
            help="Upload individual files or zip folders containing multiple files"
        )
        
        if upload_type == "Single File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'json'],
                help="Upload CSV, Excel, or JSON files (max 100MB)"
            )
            
            if uploaded_file:
                # Validate file
                is_valid, message = FileHandler.validate_file(uploaded_file)
                if is_valid:
                    st.success(" File validated successfully")
                else:
                    st.error(f" {message}")
                    uploaded_file = None
        
        else:  # Folder upload
            uploaded_file = st.file_uploader(
                "Choose a zip folder",
                type=['zip'],
                help="Upload zip files containing CSV, Excel, or JSON files (max 200MB)"
            )
            
            if uploaded_file:
                # Validate zip file
                folder_handler = FolderHandler()
                is_valid, message = folder_handler.validate_zip_file(uploaded_file)
                if is_valid:
                    st.success("Zip file validated successfully")
                else:
                    st.error(f"{message}")
                    uploaded_file = None
        
        # Processing options
        if uploaded_file:
            st.markdown("---")
            st.markdown("#### ⚙️ Processing Options")
            
            enable_ai_analysis = st.checkbox(
                "Enable AI Analysis",
                value=True,
                help="Get intelligent insights and suggestions"
            )
            
            if not enable_ai_analysis:
                st.warning("AI analysis disabled - you'll get basic quality checks only")
    
    # Main content area
    if uploaded_file is None:
        # Welcome screen
        display_welcome_screen()
    else:
        # Process uploaded file/folder
        process_uploaded_data(uploaded_file, upload_type, db, llm_client, enable_ai_analysis)

def display_welcome_screen():
    # """Display welcome screen when no file is uploaded"""

    # Create 2 columns for features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚡ Core Features")
        st.markdown("""
        #### 📄 Data Analysis
        - 📥 Support for CSV, Excel (.xlsx), and JSON files
        - 📊 Real-time quality assessment & scoring
        - 💰 Smart payment data recognition & analysis
        - 🤖 AI-powered data insights

        #### 🗂️ Batch Processing
        - 📦 Process multiple files via ZIP upload
        - 🔄 Unified quality analysis across files
        - 🔍 Cross-file pattern detection
        - 📈 Comprehensive folder summaries
        """)

    with col2:
        st.markdown("### 🔍 Quality Analysis")
        st.markdown("""
        -  **Data Profiling** - Column types and patterns
        - �🚫 **Missing Data Analysis** - Column-level gap detection
        -  **Duplicate Detection**
          - Row-level duplicates
          - Column-level duplicates
          - Value-level duplicates
        - 🔍 **Financial Analysis** - Payment status tracking
        - 🔍 **Column Similarity** - Related field detection
        """)

    # Another row with 2 columns
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🧹 Data Cleaning")
        st.markdown("""
        - 🗑️ **Duplicate Removal**
          - Remove duplicate rows
          - Remove duplicate columns
        - 📥 **Export Options**
          - CSV format
          - Excel format
        - 🧹 **Cleaning Statistics**
          - Track removed duplicates
          - Monitor data changes
        - 🔍 **Data Preview** - Before/after comparison
        """)

    with col4:
        st.markdown("### 📊 Visualization & Reporting")
        st.markdown("""
        - 🏆 **Quality Score** - Overall data health metric
        - 📈 **Interactive Charts**
          - Missing data distribution
          - Duplicate analysis graphs
          - Payment status breakdown
        - 📑 **Detailed Reports**
          - AI analysis PDF export
          - Payment status reports
          - Quality assessment summary
        """)

    st.markdown("---")

    # Get started message
    """
    **Get started by uploading a data file or zip folder using the sidebar! **
    """
    
    # Sample datasets section
    st.markdown("### 📚 Sample Datasets")
    st.markdown("Don't have data to test? Try these sample scenarios:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🛍️ Retail Sales Data**
        - Customer transactions
        - Leads form
        - Common issues: Missing customer IDs, duplicate orders
        """)
        try:
            retail_sample_path = os.path.join(current_dir, "data", "sample", "retail", "retail_sample.csv")
            with open(retail_sample_path, "rb") as file:
                retail_data = file.read()
            button_html = f"""
                <a href="data:application/octet-stream;base64,{base64.b64encode(retail_data).decode()}" 
                    download="retail_sample.csv" 
                    style="
                        display:inline-block;
                        border:1px solid #ff4d4d;
                        color:#ff4d4d;
                        padding:8px 16px;
                        border-radius:6px;
                        text-decoration:none;
                        font-family:sans-serif;
                        font-size:14px;
                    ">
                    📥 Download Retail Sample
                </a>
                """

            st.markdown(button_html, unsafe_allow_html=True)
        except Exception as e:
            st.error("Sample file not available. Please check back later.")
    
    with col2:
        st.markdown("""
        **👥 HR Employee Data**
        - Employee records
        - Department information
        - Common issues: Inconsistent job titles, missing contact info
        """)
        try:
            hr_sample_path = os.path.join(current_dir, "data", "sample", "hr", "hr_sample.csv")
            with open(hr_sample_path, "rb") as file:
                hr_data = file.read()
            button_html = f"""
                <a href="data:application/octet-stream;base64,{base64.b64encode(hr_data).decode()}" 
                    download="hr_sample.csv" 
                    style="
                        display:inline-block;
                        border:1px solid #ff4d4d;
                        color:#ff4d4d;
                        padding:8px 16px;
                        border-radius:6px;
                        text-decoration:none;
                        font-family:sans-serif;
                        font-size:14px;
                    ">
                    📥 Download HR Sample
                </a>
                """

            st.markdown(button_html, unsafe_allow_html=True)
        except Exception as e:
            st.error("Sample file not available. Please check back later.")
    
    with col3:
        st.markdown("""
        **💰 Financial Reports**
        - Transaction logs
        - Account balances
        - Common issues: Formatting inconsistencies, outlier amounts
        """)
        try:
            financial_sample_path = os.path.join(current_dir, "data", "sample", "financial", "financial_sample.csv")
            with open(financial_sample_path, "rb") as file:
                financial_data = file.read()
                button_html = f"""
                <a href="data:application/octet-stream;base64,{base64.b64encode(financial_data).decode()}" 
                    download="financial_sample.csv" 
                    style="
                        display:inline-block;
                        border:1px solid #ff4d4d;
                        color:#ff4d4d;
                        padding:8px 16px;
                        border-radius:6px;
                        text-decoration:none;
                        font-family:sans-serif;
                        font-size:14px;
                    ">
                    📥 Download Financial Sample
                </a>
                """

            st.markdown(button_html, unsafe_allow_html=True)
        except Exception as e:
            st.error("Sample file not available. Please check back later.")

    st.markdown("---")


def process_uploaded_data(uploaded_file, upload_type, db, llm_client, enable_ai_analysis):
    """Process the uploaded data and display results"""
    st.markdown(f"## Analysis Results for: {uploaded_file.name}")
    
    # Processing indicator
    with st.spinner("Processing your data..."):
        if upload_type == "Single File":
            # Process single file
            try:
                # Parse file and run quality checks
                df, metadata = FileHandler.parse_file(uploaded_file)
                
                # Save to database
                dataset_id = db.save_dataset(
                    name=metadata['filename'],
                    file_path=f"uploads/{metadata['filename']}",
                    file_size=metadata['file_size'],
                    row_count=metadata['row_count'],
                    column_count=metadata['column_count'],
                    file_type=metadata['file_type'],
                    is_folder=False
                )
                
                # Run quality checks
                quality_checker = DataQualityChecker(df)
                quality_results = quality_checker.check_data_quality()
                
                # Get AI insights only if enabled
                ai_insights = None
                if enable_ai_analysis and llm_client:
                    ai_insights = llm_client.analyze_data_quality(df, quality_results, metadata)
                    if ai_insights:
                        ai_insights['dataframe'] = df
                        # Save AI insights to database
                        db.save_ai_insight(
                            dataset_id=dataset_id,
                            issue_summary=ai_insights.get('summary', ''),
                            business_impact=ai_insights.get('business_impact', {}),
                            suggested_fixes=ai_insights.get('suggestions', {}),
                            confidence_score=ai_insights.get('confidence', 0.8)
                        )
                
                results = {
                    'dataset_id': dataset_id,
                    'dataframe': df,
                    'metadata': metadata,
                    'quality_results': quality_results,
                    'ai_insights': ai_insights,
                    'processing_status': 'success'
                }
                
                st.success("File processed successfully!")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(results['dataframe'].head(10), use_container_width=True)
                
                # Show quality dashboard
                create_quality_dashboard(results, is_folder=False)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
        
        else:  # Folder processing
            try:
                folder_handler = FolderHandler()
                processed_files, folder_metadata = folder_handler.extract_and_process_zip(uploaded_file)
                
                # Save folder as parent dataset
                parent_dataset_id = db.save_dataset(
                    name=folder_metadata['folder_name'],
                    file_path=f"uploads/{folder_metadata['folder_name']}",
                    file_size=folder_metadata['folder_size'],
                    row_count=0,
                    column_count=0,
                    file_type='.zip',
                    is_folder=True
                )
                
                folder_results = []
                
                # Process each file
                for file_data in processed_files:
                    df = file_data['dataframe']
                    metadata = file_data['metadata']
                    
                    # Save individual file
                    dataset_id = db.save_dataset(
                        name=metadata['filename'],
                        file_path=f"uploads/{folder_metadata['folder_name']}/{file_data['file_path']}",
                        file_size=metadata['file_size'],
                        row_count=metadata['row_count'],
                        column_count=metadata['column_count'],
                        file_type=metadata['file_type'],
                        is_folder=False,
                        parent_dataset_id=parent_dataset_id
                    )
                    
                    # Run quality checks
                    quality_checker = DataQualityChecker(df)
                    quality_results = quality_checker.check_data_quality()
                    
                    # Get AI insights only if enabled
                    ai_insights = None
                    if enable_ai_analysis and llm_client:
                        ai_insights = llm_client.analyze_data_quality(df, quality_results, metadata)
                        if ai_insights:
                            ai_insights['dataframe'] = df
                            # Save AI insights to database
                            db.save_ai_insight(
                                dataset_id=dataset_id,
                                issue_summary=ai_insights.get('summary', ''),
                                business_impact=ai_insights.get('business_impact', ''),
                                suggested_fixes=ai_insights.get('suggestions', {}),
                                confidence_score=ai_insights.get('confidence', 0.8)
                            )
                    
                    folder_results.append({
                        'dataset_id': dataset_id,
                        'dataframe': df,
                        'metadata': metadata,
                        'quality_results': quality_results,
                        'ai_insights': ai_insights,
                        'file_path': file_data['file_path']
                    })
                
                # Get folder summary
                folder_summary = folder_handler.get_folder_summary(processed_files)
                
                results = {
                    'parent_dataset_id': parent_dataset_id,
                    'folder_metadata': folder_metadata,
                    'folder_summary': folder_summary,
                    'file_results': folder_results,
                    'processing_status': 'success'
                }
                
                if results['processing_status'] == 'success':
                    st.success("Folder processed successfully!")
                    # Show folder dashboard
                    create_quality_dashboard(results, is_folder=True)
                else:
                    st.error(f"Processing failed: {results.get('error_message', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
