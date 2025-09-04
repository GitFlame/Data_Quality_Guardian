import pandas as pd
from typing import Dict, List, Any
import numpy as np

class DataQualityChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.quality_metrics = {}
        
    def check_data_quality(self) -> Dict[str, Any]:
        """
        Analyze the data quality of the DataFrame
        
        Returns:
            Dict[str, Any]: Dictionary containing quality metrics
        """
        metrics = {
            'row_count': len(self.df),
            'column_count': len(self.df.columns),
            'missing_data': self._check_missing_values(),
            'data_types': self._get_data_types(),
            'duplicates': self._check_duplicates(),
            'basic_stats': self._get_basic_stats(),
            'consistency': {},  # Added for compatibility
            'summary': self._generate_summary(),
            'dataframe': self.df  # Include the DataFrame for cleaning operations
        }
        
        self.quality_metrics = metrics
        return metrics
    
    def _check_missing_values(self) -> Dict[str, Any]:
        """Calculate missing values percentage for each column"""
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        total_cells = len(self.df) * len(self.df.columns)
        missing_percent = (missing / len(self.df)) * 100
        
        by_column = {
            col: {
                'missing_count': int(count),
                'missing_percentage': float(pct)
            }
            for col, count, pct in zip(self.df.columns, missing, missing_percent)
        }
        
        return {
            'total_missing_cells': int(total_missing),
            'total_cells': int(total_cells),
            'missing_data_percentage': float((total_missing / total_cells) * 100 if total_cells > 0 else 0),
            'by_column': by_column
        }
    
    def _get_data_types(self) -> Dict[str, str]:
        """Get data types for each column"""
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}
    
    def _check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicates at row, column, and value levels"""
        # Row-level duplicates
        total_rows = len(self.df)
        
        # Find all duplicated rows (excluding the first occurrence)
        duplicates_without_first = self.df.duplicated(keep='first')
        duplicate_rows_to_remove = duplicates_without_first.sum()
        
        # Find all rows that are part of duplicate sets (including first occurrences)
        all_duplicate_rows = self.df.duplicated(keep=False)
        total_rows_in_duplicate_sets = all_duplicate_rows.sum()
        
        duplicate_rows_percentage = (duplicate_rows_to_remove / total_rows) * 100 if total_rows > 0 else 0

        # Get actual duplicate row content for reference
        duplicate_row_examples = []
        if total_rows_in_duplicate_sets > 0:
            # Group duplicate rows and get examples
            grouped_dups = self.df[all_duplicate_rows].groupby(list(self.df.columns))
            for _, group in grouped_dups:
                if len(group) > 1:  # Only actual duplicates
                    duplicate_row_examples.append({
                        'row_content': group.iloc[0].to_dict(),
                        'occurrence_count': len(group),
                        'row_indices': group.index.tolist()
                    })

        # Column-level duplicates with improved detection
        columns_df = self.df.astype(str)  # Convert to string to compare content
        duplicate_columns = {}
        column_similarity = {}  # Track partial column similarities

        for i, col1 in enumerate(columns_df.columns):
            for j, col2 in enumerate(columns_df.columns[i+1:], start=i+1):
                # Check for exact matches
                if columns_df[col1].equals(columns_df[col2]):
                    if col1 not in duplicate_columns:
                        duplicate_columns[col1] = []
                    duplicate_columns[col1].append({
                        'column': col2,
                        'match_type': 'exact',
                        'similarity': 100.0
                    })
                else:
                    # Check for high similarity (optional)
                    overlap = (columns_df[col1] == columns_df[col2]).mean() * 100
                    if overlap >= 80:  # 80% or more similarity
                        if col1 not in column_similarity:
                            column_similarity[col1] = []
                        column_similarity[col1].append({
                            'column': col2,
                            'match_type': 'similar',
                            'similarity': round(overlap, 2)
                        })

        # Value-level duplicates within columns with improved analysis
        column_duplicates = {}
        for column in self.df.columns:
            value_counts = self.df[column].value_counts()
            duplicates = value_counts[value_counts > 1]
            if not duplicates.empty:
                # Calculate percentage of rows affected by duplicates
                total_duplicate_rows = sum(duplicates)
                duplicate_percentage = (total_duplicate_rows / len(self.df)) * 100
                
                column_duplicates[column] = {
                    'duplicate_count': len(duplicates),
                    'total_duplicate_rows': int(total_duplicate_rows),
                    'duplicate_percentage': round(duplicate_percentage, 2),
                    'duplicate_values': {
                        str(value): {
                            'count': int(count),
                            'percentage': round((count / len(self.df)) * 100, 2)
                        }
                        for value, count in duplicates.items()
                    }
                }

        # Overall severity assessment
        if duplicate_rows_to_remove == 0 and not duplicate_columns and not column_duplicates:
            severity = "Good"
        elif duplicate_rows_percentage < 5 and len(duplicate_columns) == 0:
            severity = "Low"
        elif duplicate_rows_percentage < 10 and len(duplicate_columns) <= 1:
            severity = "Medium"
        else:
            severity = "High"

        # Calculate unique rows (rows that should remain after cleaning)
        unique_rows = total_rows - duplicate_rows_to_remove

        return {
            'row_duplicates': {
                'total_duplicates': total_rows_in_duplicate_sets,  # Total rows that are duplicates
                'removable_duplicates': duplicate_rows_to_remove,  # Duplicates that can be removed
                'unique_rows': unique_rows,  # Rows that will remain after cleaning
                'total_rows': total_rows,
                'duplicate_percentage': duplicate_rows_percentage,
                'duplicate_examples': duplicate_row_examples[:5]  # Show up to 5 examples
            },
            'column_duplicates': {
                'duplicate_columns': duplicate_columns,
                'similar_columns': column_similarity,
                'total_duplicate_columns': sum(len(dups) for dups in duplicate_columns.values()),
                'total_similar_columns': sum(len(sims) for sims in column_similarity.values())
            },
            'value_duplicates': column_duplicates,
            'severity': severity
        }
    
    def _get_basic_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }
            
        return stats
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the dataset"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = self.df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
        
        # Calculate total memory usage in bytes
        memory_usage = self.df.memory_usage(deep=True).sum()
        
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'total_cells': len(self.df) * len(self.df.columns),
            'memory_usage': memory_usage  # Memory usage in bytes
        }
