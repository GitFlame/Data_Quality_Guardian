import zipfile
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile
import shutil
from src.ingestion.file_handler import FileHandler

class FolderHandler:
    """Handles folder/zip file upload and processing"""
    
    def __init__(self):
        self.file_handler = FileHandler()
        self.temp_extract_dir = None
    
    def validate_zip_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded zip file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        if not uploaded_file.name.lower().endswith('.zip'):
            return False, "File must be a zip archive"
        
        if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit for zip
            return False, "Zip file too large (max 200MB)"
        
        return True, "Zip file valid"
    
    def extract_and_process_zip(self, uploaded_zip) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract zip file and process all supported files inside"""
        results = []
        folder_metadata = {
            'folder_name': uploaded_zip.name,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'supported_files': [],
            'unsupported_files': [],
            'folder_size': uploaded_zip.size
        }
        
        # Create temporary directory for extraction
        self.temp_extract_dir = tempfile.mkdtemp()
        
        try:
            # Extract zip file
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(self.temp_extract_dir)
            
            # Find all files recursively
            all_files = []
            for root, dirs, files in os.walk(self.temp_extract_dir):
                for file in files:
                    if not file.startswith('.'):  # Skip hidden files
                        all_files.append(os.path.join(root, file))
            
            folder_metadata['total_files'] = len(all_files)
            
            # Process each file
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension in ['.csv', '.xlsx', '.json']:
                    folder_metadata['supported_files'].append(file_name)
                    
                    try:
                        # Read file and process
                        with open(file_path, 'rb') as f:
                            # Create a file-like object for FileHandler
                            class MockUploadedFile:
                                def __init__(self, file_path, file_obj):
                                    self.name = os.path.basename(file_path)
                                    self.size = os.path.getsize(file_path)
                                    self._file = file_obj
                                
                                def read(self, size=-1):
                                    return self._file.read(size)
                                
                                def seek(self, pos):
                                    return self._file.seek(pos)
                            
                            mock_file = MockUploadedFile(file_path, f)
                            df, file_metadata = self.file_handler.parse_file(mock_file)
                            
                            # Add relative path info
                            relative_path = os.path.relpath(file_path, self.temp_extract_dir)
                            file_metadata['relative_path'] = relative_path
                            file_metadata['extracted_from'] = uploaded_zip.name
                            
                            results.append({
                                'dataframe': df,
                                'metadata': file_metadata,
                                'file_path': relative_path
                            })
                            
                            folder_metadata['processed_files'] += 1
                            
                    except Exception as e:
                        folder_metadata['failed_files'] += 1
                        print(f"Failed to process {file_name}: {str(e)}")
                
                else:
                    folder_metadata['unsupported_files'].append(file_name)
            
            return results, folder_metadata
            
        except Exception as e:
            raise Exception(f"Error processing zip file: {str(e)}")
        
        finally:
            # Cleanup temporary directory
            if self.temp_extract_dir and os.path.exists(self.temp_extract_dir):
                shutil.rmtree(self.temp_extract_dir)
    
    def get_folder_summary(self, processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for all files in folder"""
        if not processed_files:
            return {}
        
        total_rows = sum(file_data['metadata']['row_count'] for file_data in processed_files)
        total_columns = sum(file_data['metadata']['column_count'] for file_data in processed_files)
        total_size = sum(file_data['metadata']['file_size'] for file_data in processed_files)
        
        file_types = {}
        for file_data in processed_files:
            file_type = file_data['metadata']['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_files_processed': len(processed_files),
            'total_rows_across_files': total_rows,
            'total_columns_across_files': total_columns,
            'total_size_bytes': total_size,
            'file_type_distribution': file_types,
            'average_rows_per_file': total_rows // len(processed_files) if processed_files else 0
        }