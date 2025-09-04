from datetime import datetime
import json
import os
from typing import Dict, Any, List
from pathlib import Path

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

if not SQLALCHEMY_AVAILABLE:
    raise ImportError(
        "SQLAlchemy is required but not installed. "
        "Please install it with: pip install sqlalchemy"
    )

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500))
    upload_time = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    row_count = Column(Integer)
    column_count = Column(Integer)
    file_type = Column(String(50))
    status = Column(String(50), default="uploaded")
    is_folder = Column(Boolean, default=False)  # New field for folder processing
    parent_dataset_id = Column(Integer)  # For linking files in a folder

class QualityCheck(Base):
    __tablename__ = "quality_checks"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer)
    check_type = Column(String(100))
    severity = Column(String(50))
    affected_rows = Column(Integer)
    affected_columns = Column(Text)  # JSON string
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class AIInsight(Base):
    __tablename__ = "ai_insights"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer)
    issue_summary = Column(Text)
    business_impact = Column(Text)
    suggested_fixes = Column(Text)  # JSON string
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        return self.SessionLocal()
    
    def save_dataset(self, name: str, file_path: str, file_size: int, 
                    row_count: int, column_count: int, file_type: str,
                    is_folder: bool = False, parent_dataset_id: int = None) -> int:
        session = self.get_session()
        try:
            dataset = Dataset(
                name=name,
                file_path=file_path,
                file_size=file_size,
                row_count=row_count,
                column_count=column_count,
                file_type=file_type,
                is_folder=is_folder,
                parent_dataset_id=parent_dataset_id
            )
            session.add(dataset)
            session.commit()
            return dataset.id
        finally:
            session.close()
    
    def save_quality_check(self, dataset_id: int, check_type: str, 
                          severity: str, affected_rows: int,
                          affected_columns: List[str], description: str):
        session = self.get_session()
        try:
            quality_check = QualityCheck(
                dataset_id=dataset_id,
                check_type=check_type,
                severity=severity,
                affected_rows=affected_rows,
                affected_columns=json.dumps(affected_columns),
                description=description
            )
            session.add(quality_check)
            session.commit()
        finally:
            session.close()
    
    def save_ai_insight(self, dataset_id: int, issue_summary: str,
                       business_impact: Any, suggested_fixes: Dict[str, Any],
                       confidence_score: float):
        session = self.get_session()
        try:
            # Convert dictionaries to JSON strings
            if isinstance(business_impact, dict):
                business_impact = json.dumps(business_impact)
            if isinstance(suggested_fixes, dict):
                suggested_fixes = json.dumps(suggested_fixes)
            
            insight = AIInsight(
                dataset_id=dataset_id,
                issue_summary=issue_summary,
                business_impact=business_impact,
                suggested_fixes=suggested_fixes,
                confidence_score=confidence_score
            )
            session.add(insight)
            session.commit()
            return insight.id
        finally:
            session.close()

    def get_dataset_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent dataset analysis history"""
        session = self.get_session()
        try:
            datasets = (session.query(Dataset)
                       .order_by(Dataset.upload_time.desc())
                       .limit(limit)
                       .all())
            
            history = []
            for dataset in datasets:
                # Get associated AI insights
                insight = (session.query(AIInsight)
                         .filter(AIInsight.dataset_id == dataset.id)
                         .first())
                
                history.append({
                    'id': dataset.id,
                    'name': dataset.name,
                    'upload_time': dataset.upload_time,
                    'file_type': dataset.file_type,
                    'row_count': dataset.row_count,
                    'column_count': dataset.column_count,
                    'is_folder': dataset.is_folder,
                    'quality_summary': insight.issue_summary if insight else None,
                    'confidence_score': insight.confidence_score if insight else None
                })
            
            return history
        finally:
            session.close()

    def get_dataset_details(self, dataset_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific dataset"""
        session = self.get_session()
        try:
            dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                return None
            
            # Get AI insights
            insight = (session.query(AIInsight)
                      .filter(AIInsight.dataset_id == dataset_id)
                      .first())
            
            # Get quality checks
            quality_checks = (session.query(QualityCheck)
                            .filter(QualityCheck.dataset_id == dataset_id)
                            .all())
            
            return {
                'dataset': {
                    'id': dataset.id,
                    'name': dataset.name,
                    'upload_time': dataset.upload_time,
                    'file_type': dataset.file_type,
                    'row_count': dataset.row_count,
                    'column_count': dataset.column_count,
                    'file_size': dataset.file_size,
                    'is_folder': dataset.is_folder,
                    'parent_dataset_id': dataset.parent_dataset_id
                },
                'ai_insights': {
                    'summary': insight.issue_summary if insight else None,
                    'business_impact': json.loads(insight.business_impact) if insight and insight.business_impact else None,
                    'suggested_fixes': json.loads(insight.suggested_fixes) if insight and insight.suggested_fixes else None,
                    'confidence_score': insight.confidence_score if insight else None
                } if insight else None,
                'quality_checks': [{
                    'check_type': check.check_type,
                    'severity': check.severity,
                    'affected_rows': check.affected_rows,
                    'affected_columns': json.loads(check.affected_columns) if check.affected_columns else [],
                    'description': check.description,
                    'created_at': check.created_at
                } for check in quality_checks]
            }
        finally:
            session.close()

    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over time"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            from datetime import datetime, timedelta
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get average confidence scores over time
            confidence_trend = (session.query(
                func.date(AIInsight.created_at).label('date'),
                func.avg(AIInsight.confidence_score).label('avg_confidence')
            ).filter(AIInsight.created_at >= start_date)
             .group_by(func.date(AIInsight.created_at))
             .all())
            
            # Get severity distributions
            severity_dist = (session.query(
                QualityCheck.severity,
                func.count(QualityCheck.id).label('count')
            ).filter(QualityCheck.created_at >= start_date)
             .group_by(QualityCheck.severity)
             .all())
            
            return {
                'confidence_trend': [
                    {'date': str(date), 'confidence': float(conf)}
                    for date, conf in confidence_trend
                ],
                'severity_distribution': [
                    {'severity': sev, 'count': int(count)}
                    for sev, count in severity_dist
                ]
            }
        finally:
            session.close()

    def get_folder_analysis(self, parent_dataset_id: int) -> Dict[str, Any]:
        """Get analysis for all files in a folder"""
        session = self.get_session()
        try:
            parent = session.query(Dataset).filter(Dataset.id == parent_dataset_id).first()
            if not parent or not parent.is_folder:
                return None
            
            # Get all files in the folder
            files = (session.query(Dataset)
                    .filter(Dataset.parent_dataset_id == parent_dataset_id)
                    .all())
            
            file_analyses = []
            for file in files:
                # Get insights for each file
                insight = (session.query(AIInsight)
                         .filter(AIInsight.dataset_id == file.id)
                         .first())
                
                file_analyses.append({
                    'file_name': file.name,
                    'row_count': file.row_count,
                    'column_count': file.column_count,
                    'quality_summary': insight.issue_summary if insight else None,
                    'confidence_score': insight.confidence_score if insight else None
                })
            
            return {
                'folder_name': parent.name,
                'upload_time': parent.upload_time,
                'file_count': len(files),
                'files': file_analyses
            }
        finally:
            if session:
                session.close()