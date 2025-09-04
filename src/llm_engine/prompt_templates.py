"""
Prompt templates for LLM interactions in the data quality guardian.
"""

import json
from typing import Dict, Any

class PromptTemplates:
    """Collection of prompt templates for LLM interactions"""
    
    @staticmethod
    def data_analysis_prompt(data_info: Dict[str, Any]) -> str:
        """Generate prompt for data analysis"""
        return f"""
        Analyze the following dataset information and provide insights:
        
        Dataset Overview:
        - Number of rows: {data_info.get('row_count', 'N/A')}
        - Number of columns: {data_info.get('column_count', 'N/A')}
        
        Column Data Types:
        {data_info.get('data_types', {})}
        
        Missing Values:
        {data_info.get('missing_values', {})}
        
        Please provide:
        1. A summary of data quality issues
        2. Recommendations for data cleaning
        3. Potential insights from the data structure
        4. Suggestions for data validation rules
        """

    @staticmethod
    def data_validation_prompt(validation_results: Dict[str, Any]) -> str:
        """Generate prompt for data validation insights"""
        return f"""
        Review these data validation results and provide insights:
        
        Validation Results:
        {validation_results}
        
        Please provide:
        1. Analysis of validation failures
        2. Impact assessment of data quality issues
        3. Recommendations for data quality improvement
        4. Suggested validation thresholds
        """

    @staticmethod
    def get_quality_score_prompt(metrics: Dict[str, Any]) -> str:
        """Generate prompt for calculating quality score"""
        return f"""
        Based on these quality metrics:
        {metrics}
        
        Please:
        1. Calculate an overall data quality score (0-100)
        2. Explain the scoring methodology
        3. Identify critical quality issues
        4. Suggest quality improvement priorities
        """

    @staticmethod
    def anomaly_detection_prompt(data_stats: Dict[str, Any]) -> str:
        """Generate prompt for anomaly detection"""
        return f"""
        Analyze these statistical measures for anomalies:
        {data_stats}
        
        Please identify:
        1. Statistical outliers
        2. Unusual patterns or distributions
        3. Potential data entry errors
        4. Recommendations for handling anomalies
        """

    @staticmethod
    def improvement_suggestions_prompt(quality_analysis: Dict[str, Any]) -> str:
        """Generate prompt for improvement suggestions"""
        return f"""
        Based on this quality analysis:
        {quality_analysis}
        
        Please provide:
        1. Specific data quality improvement actions
        2. Priority order for improvements
        3. Expected impact of each improvement
        4. Implementation recommendations
        """

    @staticmethod
    def get_domain_detection_prompt(data_summary: Dict[str, Any]) -> str:
        """Generate prompt to detect data domain and role"""
        columns_info = data_summary.get('columns_info', {})
        metadata = data_summary.get('metadata', {})
        
        return f"""You are a data domain expert. Analyze the file name, column names, and data patterns to determine the type of data.

FILE INFORMATION:
- Filename: {metadata.get('filename', 'N/A')}
- Total Columns: {metadata.get('columns', 'N/A')}
- Column Names and Types: {json.dumps(columns_info, indent=2)}

Determine the domain and provide response in JSON format:
{{
    "domain": "The primary domain (e.g., Financial, HR, Sales, Client Management, Project Management)",
    "sub_domain": "More specific classification (e.g., Payment Processing, Employee Records, Sales Pipeline)",
    "confidence": "Confidence score between 0-1",
    "key_indicators": ["List of columns/patterns that indicate this domain"],
    "critical_fields": ["List of most important fields for this domain"],
    "expected_patterns": {{
        "field_name": "Expected data pattern or format"
    }}
}}

Consider common domains like:
1. Financial Data (payments, invoices, transactions)
2. HR Data (employee records, performance, attendance)
3. Sales Data (revenue, customer info, product sales)
4. Client Management (client details, project status)
5. Operations Data (inventory, logistics)
6. Marketing Data (campaign metrics, leads)
"""

    @staticmethod
    def get_quality_analysis_prompt(data_summary: Dict[str, Any], domain_info: Dict[str, Any] = None) -> str:
        """Generate comprehensive prompt for data quality analysis with Gemini API"""
        missing_data = data_summary.get('quality_issues', {}).get('missing_data', {})
        duplicates = data_summary.get('quality_issues', {}).get('duplicates', {})
        columns_info = data_summary.get('columns_info', {})
        metadata = data_summary.get('metadata', {})
        
        domain_context = ""
        if domain_info:
            domain_context = f"""
DOMAIN CONTEXT:
- Primary Domain: {domain_info.get('domain')}
- Sub-domain: {domain_info.get('sub_domain')}
- Critical Fields: {', '.join(domain_info.get('critical_fields', []))}
"""
        
        response_template = '''
{
    "summary": "A domain-specific summary of the main data quality issues",
    "domain_insights": {
        "critical_issues": ["List of domain-specific critical issues found"],
        "compliance_concerns": ["Any compliance or regulatory issues based on domain"],
        "data_patterns": ["Notable patterns or anomalies specific to this domain"]
    },
    "business_impact": {
        "operational": "How these issues affect day-to-day operations",
        "financial": "Potential financial impact of the issues",
        "compliance": "Any compliance or regulatory risks",
        "stakeholder": "Impact on key stakeholders"
    },
    "suggestions": {
        "high_priority": ["List of urgent domain-specific fixes needed and corrective actions"],
        "medium_priority": ["List of important but not urgent improvements and corrective actions"],
        "low_priority": ["List of minor enhancements and fixes and corrective actions"],
        "validation_rules": ["Suggested domain-specific validation rules"]
    },
    "domain_specific_metrics": {
        "key_metrics": {"metric_name": "current_status"},
        "benchmarks": {"metric_name": "expected_value"}
    },
    "confidence": "A number between 0 and 1 indicating your confidence in this analysis"
}
'''
        
        # Build the prompt parts separately to avoid f-string formatting issues
        overview = f"""You are a data quality expert specializing in {domain_info.get('domain', 'data')} analysis.
Based on your domain expertise, provide a detailed quality assessment.

DATA OVERVIEW:
- Filename: {metadata.get('filename', 'N/A')}
- Total Rows: {metadata.get('rows', 'N/A')}
- Total Columns: {metadata.get('columns', 'N/A')}
- File Format: {metadata.get('file_type', 'N/A')}
{domain_context}"""

        metrics = f"""
QUALITY METRICS:
1. Missing Data:
   - Total Missing Values: {missing_data.get('total_missing_cells', 0)}
   - Overall Missing Percentage: {missing_data.get('missing_data_percentage', 0):.2f}%

2. Duplicates:
   - Duplicate Rows: {duplicates.get('total_duplicates', 0)}
   - Duplication Rate: {duplicates.get('duplicate_percentage', 0):.2f}%
   - Data Quality Severity: {duplicates.get('severity', 'Unknown')}"""

        column_types = f"""
3. Column Types:
{json.dumps(columns_info, indent=2)}"""

        instructions = """
Analyze this information and provide a response in the following JSON format:"""

        footer = """
Note: Keep your response focused on actionable insights and practical recommendations. Base confidence score on the comprehensiveness of the available data."""

        # Combine all parts
        prompt = f"{overview}{metrics}{column_types}{instructions}\n{response_template}{footer}"
        
        return prompt
