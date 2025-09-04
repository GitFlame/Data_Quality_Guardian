AI-Powered Data Quality Analysis & Monitoring System

## 🚀 Features

### Core Capabilities

- **Single File Analysis** - Upload CSV, Excel, or JSON files
- **Folder Processing** - Batch analyze multiple files from zip uploads
- **AI-Powered Insights** - Intelligent analysis using Gemini 2.0 Flash
- **Interactive Dashboard** - Real-time visualizations and reports
- **Quality Scoring** - Comprehensive health metrics (0-100 scale)

### Quality Checks

- **Missing Data Detection** - Find and analyze gaps
- **Duplicate Identification** - Spot redundant records
- **Data Type Validation** - Ensure proper types
- **Consistency Analysis** - Check formatting standards
- **Outlier Detection** - Identify unusual values
- **Pattern Recognition** - AI-powered anomaly detection

## 🛠️ Tech Stack

- **Backend**: Python 3.12.1
- **LLM**: Google Gemini 2.0 Flash (via LangChain)
- **Frontend**: Streamlit
- **Database**: SQLite (development) 
- **Data Processing**: pandas, openpyxl, python-docx
- **Visualization**: Plotly, matplotlib
- **AI Framework**: LangChain

## 🚦 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd data_quality_guardian
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Add your API keys to .env
#For Google api key go to the google AI studio to create API key
#URL: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run Application

```bash
streamlit run app.py
```

## 📊 Usage

### Single File Analysis

1. Upload CSV/Excel/JSON file
2. Get instant quality report
3. View AI-powered insights
4. Download detailed analysis

### Folder Processing

1. Upload zip file with multiple data files
2. Batch process all supported files
3. Get combined analysis report
4. Compare quality across files

## 🏗️ Project Structure

```
data_quality_guardian/
├── src/
│   ├── config/settings.py          # Configuration management
│   ├── database/models.py          # Database models & manager
│   ├── ingestion/
│   │   ├── file_handler.py         # Single file processing
│   │   └── folder_handler.py       # Zip folder processing
│   ├── analysis/quality_checker.py # Core quality analysis
│   ├── llm_engine/
│   │   ├── llm_client.py          # LLM integration
│   │   └── prompt_templates.py     # AI prompts
│   └── dashboard/components.py     # UI components
├── data/sample/                    # Sample datasets
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── app.py                         # Main Streamlit app
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🎯 Quality Scoring Algorithm

**Score Range**: 0-100 points

**Deduction Rules**:

- Missing Data: -0.5 points per 1% (max -30 points)
- Duplicates: -0.3 points per 1% (max -20 points)
- Consistency Issues: -2 points per issue (max -15 points)
- Data Type Issues: -5 points per major issue (max -20 points)
- Outliers: -0.1 points per 1% (max -10 points)

**Grade Scale**:

- 🟢 90-100: Excellent
- 🟡 70-89: Good
- 🟠 50-69: Fair
- 🔴 0-49: Poor

## 🤖 AI Analysis Features

### Smart Issue Detection

- Context-aware problem identification
- Business domain understanding
- Severity assessment with reasoning

### Actionable Recommendations

- Specific steps to fix issues
- Implementation difficulty scoring
- Prevention strategies

### Business Impact Assessment

- Real-world consequence analysis
- Risk evaluation
- Compliance considerations

## 🔧 Development

### VS Code Setup

1. Install recommended extensions:

   - Python (Microsoft)
   - Pylance
   - SQLite Viewer
   - CSV Rainbow

2. Configure debugger (launch.json provided)

3. Use integrated terminal for development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/  # Format code
flake8 src/ # Lint code
```

## 🚀 Deployment Options

### Local Development

```bash
streamlit run app.py
```

### Docker Deployment

```bash
docker build -t data-quality-guardian .
docker run -p 8501:8501 data-quality-guardian
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.
