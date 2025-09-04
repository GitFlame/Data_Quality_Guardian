import os
from typing import Dict, Any, Optional
import json
import datetime
import pandas as pd

# Conditional imports to handle potential missing dependencies gracefully
try:
    import google.generativeai as genai
    from langchain.llms import GooglePalm
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from src.config.settings import settings
from src.llm_engine.prompt_templates import PromptTemplates


class LLMClient:
    """LLM client for data quality analysis"""

    def __init__(self, provider="gemini"):
        self.provider = provider.lower()
        self.prompt_templates = PromptTemplates()

        if self.provider == "gemini":
            if not DEPENDENCIES_AVAILABLE:
                raise ImportError(
                    "Required LLM dependencies not found. Please install with: "
                    "pip install google-generativeai langchain"
                )
            self.api_key = settings.GOOGLE_API_KEY
            if not self.api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file")

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

        elif self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not found. Please install with: pip install openai")

            self.api_key = settings.OPENAI_API_KEY
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

            #  Correct: keep client, don’t overwrite with raw module
            self.openai = OpenAI(api_key=self.api_key)

       

        else:
            raise ValueError("Unsupported LLM provider")

    def analyze_data_quality(
        self, df: pd.DataFrame, quality_results: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze data quality using LLM with domain-specific insights"""
        try:
            print("Starting data quality analysis...")

            # Prepare data summary
            data_summary = self._prepare_data_summary(df, quality_results, metadata)
            print("Data summary prepared successfully")

            # Step 1: Detect domain
            print("Detecting data domain...")
            domain_prompt = self.prompt_templates.get_domain_detection_prompt(data_summary)

            if self.provider == "gemini":
                domain_response = self.model.generate_content(domain_prompt)
                if not domain_response or not domain_response.text:
                    raise ValueError("Empty domain detection response from Gemini API")
                domain_info = self._parse_llm_response(domain_response.text)

                print(
                    f"Detected domain: {domain_info.get('domain')} with confidence: {domain_info.get('confidence')}"
                )

                analysis_prompt = self.prompt_templates.get_quality_analysis_prompt(data_summary, domain_info)
                analysis_response = self.model.generate_content(analysis_prompt)
                if not analysis_response or not analysis_response.text:
                    raise ValueError("Empty analysis response from Gemini API")

                quality_insights = self._parse_llm_response(analysis_response.text)

            elif self.provider == "openai":
                completion = self.openai.chat.completions.create(
                    model="gpt-5",  
                    messages=[{"role": "user", "content": domain_prompt}],
                )
                domain_text = completion.choices[0].message.content
                domain_info = self._parse_llm_response(domain_text)

                print(
                    f"Detected domain: {domain_info.get('domain')} with confidence: {domain_info.get('confidence')}"
                )

                analysis_prompt = self.prompt_templates.get_quality_analysis_prompt(data_summary, domain_info)
                completion = self.openai.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": analysis_prompt}],
                )
                analysis_text = completion.choices[0].message.content
                quality_insights = self._parse_llm_response(analysis_text)

            

            else:
                raise ValueError("Unsupported LLM provider")

            if not quality_insights:
                raise ValueError("Failed to parse quality analysis response")

            final_insights = {
                "domain_info": domain_info,
                "summary": quality_insights.get("summary"),
                "domain_insights": quality_insights.get("domain_insights", {}),
                "business_impact": quality_insights.get("business_impact", {}),
                "suggestions": quality_insights.get("suggestions", {}),
                "domain_specific_metrics": quality_insights.get("domain_specific_metrics", {}),
                "confidence": quality_insights.get("confidence", 0.5),
            }

            print("Domain-specific analysis completed successfully")
            return final_insights

        except Exception as e:
            import traceback

            print(f"LLM Analysis Error: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())

            return {
                "domain_info": {
                    "domain": "Unknown",
                    "sub_domain": "Unknown",
                    "confidence": 0.0,
                    "key_indicators": [],
                },
                "summary": f"Basic analysis completed. AI analysis failed: {str(e)}",
                "domain_insights": {
                    "critical_issues": ["AI analysis failed"],
                    "compliance_concerns": [],
                    "data_patterns": [],
                },
                "business_impact": {
                    "operational": "Unable to assess operational impact due to AI service error.",
                    "financial": "Unable to assess financial impact due to AI service error.",
                    "compliance": "Unable to assess compliance impact due to AI service error.",
                    "stakeholder": "Unable to assess stakeholder impact due to AI service error.",
                },
                "suggestions": {
                    "error": "AI suggestions unavailable",
                    "high_priority": [],
                    "medium_priority": [],
                    "low_priority": [],
                },
                "domain_specific_metrics": {},
                "confidence": 0.0,
            }

    def _prepare_data_summary(
        self, df: pd.DataFrame, quality_results: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare concise data summary for LLM analysis"""

        def serialize_value(val):
            if pd.isna(val):
                return None
            if isinstance(val, (pd.Timestamp, datetime.datetime)):
                return val.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(val, "tolist"):  # numpy arrays etc
                return val.tolist()
            if isinstance(val, (int, float, str, bool, type(None))):
                return val
            return str(val)

        # Sample data
        sample_data = []
        for record in df.head(3).to_dict("records"):
            sample_data.append({k: serialize_value(v) for k, v in record.items()})

        # Column info
        columns_info = {}
        for col in df.columns:
            sample_values = df[col].dropna().head(3)
            columns_info[col] = {
                "type": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "sample_values": [serialize_value(val) for val in sample_values],
            }

        def serialize_dict(d):
            if isinstance(d, dict):
                return {k: serialize_dict(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return [serialize_dict(x) for x in d]
            else:
                return serialize_value(d)

        summary = {
            "metadata": {
                "filename": metadata["filename"],
                "rows": metadata["row_count"],
                "columns": metadata["column_count"],
                "file_type": metadata["file_type"],
            },
            "sample_data": sample_data,
            "columns_info": columns_info,
            "quality_issues": {
                "missing_data": serialize_dict(quality_results["missing_data"]),
                "duplicates": serialize_dict(quality_results["duplicates"]),
                "consistency": serialize_dict(quality_results.get("consistency", {})),
            },
        }

        return summary

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured insights"""
        try:
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                return json.loads(json_str)

            # Fallback: no JSON
            insights = {
                "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "business_impact": "Analyze the identified issues to understand potential business consequences.",
                "suggestions": {
                    "general": [
                        "Review the identified data quality issues",
                        "Implement data validation processes",
                    ]
                },
                "confidence": 0.7,
            }
            return insights

        except Exception as e:
            return {
                "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "business_impact": "Unable to parse detailed business impact.",
                "suggestions": {"parsing_error": f"Error parsing suggestions: {str(e)}"},
                "confidence": 0.5,
            }
