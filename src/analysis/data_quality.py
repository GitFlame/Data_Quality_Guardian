import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re

class DataQualityChecker:
    """Enhanced data quality checking with comprehensive analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.issues = defaultdict(list)
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess data for analysis"""
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run comprehensive quality checks"""
        results = {
            'summary': self._get_summary(),
            'missing_data': self._check_missing_data(),
            'duplicates': self._check_duplicates(),
            'data_types': self._check_data_types(),
            'outliers': self._detect_outliers(),
            'consistency': self._check_consistency(),
            'patterns': self._analyze_patterns()
        }
        return results

    def _get_summary(self) -> Dict[str, Any]:
        """Get comprehensive dataset summary"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'column_names': list(self.df.columns),
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
