import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        self.SAMPLE_DIR = self.DATA_DIR / "sample"
        
        # Create necessary directories
        self._create_directories()
        
        # LLM Configuration with defaults
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # Database configuration
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL", 
            f"sqlite:///{str(self.PROJECT_ROOT / 'data_quality.db')}"
        )
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.DATA_DIR, self.UPLOAD_DIR, self.SAMPLE_DIR]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create directory {directory}: {str(e)}")
    # Application
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # File Processing (Updated for folder support)
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    SUPPORTED_FILE_TYPES = ['.csv', '.xlsx', '.json', '.txt']
    SUPPORTED_ARCHIVE_TYPES = ['.zip']
    
    # LLM Settings
    DEFAULT_MODEL = "gemini-2.0-flash-exp"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # Quality Check Thresholds
    MISSING_DATA_HIGH_THRESHOLD = 50.0
    MISSING_DATA_MEDIUM_THRESHOLD = 20.0
    DUPLICATE_HIGH_THRESHOLD = 30.0
    DUPLICATE_MEDIUM_THRESHOLD = 10.0
    
    def __post_init__(self):
        # Create necessary directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()