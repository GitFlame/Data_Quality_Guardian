import pandas as pd
import json
from typing import Tuple, Dict, Any
import os
from pathlib import Path

class FileHandler:
    """Handles individual file upload and parsing"""
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        supported_types = ['.csv', '.xlsx', '.json', '.zip']  # Added .zip
        
        if file_extension not in supported_types:
            return False, f"Unsupported file type: {file_extension}. Supported: {', '.join(supported_types)}"
        
        max_size = 100 * 1024 * 1024  # 100MB for individual files
        if file_extension == '.zip':
            max_size = 200 * 1024 * 1024  # 200MB for zip files
        
        if uploaded_file.size > max_size:
            return False, f"File too large (max {max_size // (1024*1024)}MB)"
        
        return True, "File valid"
    
    @staticmethod
    def parse_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse uploaded file and return DataFrame with metadata"""
        file_extension = Path(uploaded_file.name).suffix.lower()
        metadata = {
            'filename': uploaded_file.name,
            'file_size': getattr(uploaded_file, 'size', 0),
            'file_type': file_extension
        }
        
        try:
            if file_extension == '.csv':
                # Try different encodings for CSV
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    
            elif file_extension == '.xlsx':
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
                
            elif file_extension == '.json':
                uploaded_file.seek(0)
                data = json.load(uploaded_file)
                
                # Convert data to DataFrame
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                else:
                    df = pd.DataFrame([data])
                
                # Convert list/dict columns to string representation
                for col in df.columns:
                    mask = df[col].apply(lambda x: isinstance(x, (list, dict)))
                    if mask.any():
                        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: json.dumps(x))
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Clean column names (remove extra spaces, special characters)
            df.columns = df.columns.str.strip()
            
            metadata.update({
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            })
            
            return df, metadata
            
        except Exception as e:
            raise Exception(f"Error parsing file {uploaded_file.name}: {str(e)}")