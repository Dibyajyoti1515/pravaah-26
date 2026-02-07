# ML Hackathon PRAVAAH - Customer Service Conversation Analysis

An advanced AI-powered system for analyzing customer service conversations, detecting critical patterns, and generating causal insights using vector databases, sentiment analysis, and LLMs. Built for the ML Hackathon PRAVAAH competition.

## 🎯 Project Overview

This system processes customer service transcripts to:
- Perform sentiment analysis on turn-level conversations
- Detect 50+ critical patterns (escalations, frustration, broken promises, etc.)
- Build semantic search capabilities using ChromaDB vector database
- Generate causal explanations using Google Gemini LLM
- Support multi-turn conversational reasoning
- Produce structured analysis outputs for reporting

## 📁 Project Structure

```
.
├── Codebase/
│   ├── app.ipynb                          # Main Jupyter notebook with complete pipeline
│   ├── requirements.txt                   # Python dependencies
│   ├── utterances_final.csv              # Processed turn-level data with sentiment
│   ├── conversation_level_summary.csv    # Conversation-level aggregated insights
│   └── chroma_db_turns3/                 # Persistent ChromaDB vector store
│
├── Output/
│   ├── task1_queries.csv                 # Batch query analysis results
│   └── Task_2.csv                        # Multi-turn conversation analysis results
│
├── preprocessing/
│   ├── conversations_overview.csv        # Initial conversation metadata
│   ├── detaild_analysis_data.ipynb      # Data exploration notebook
│   ├── google_colab.ipynb               # Google Colab preprocessing
│   └── utterances_partial_0_10000.csv   # Partial utterance data
│
├── Conversational_Transcript_Dataset.json # Raw conversation transcripts
├── ML_HACKATHON_PRAVAAH.pdf              # Competition documentation
├── utterances.csv                         # Raw turn-level utterances
└── README.md                              # This file
```

## 🚀 Key Features

### 1. Sentiment Analysis
- Turn-level sentiment scoring using transformer models
- Sentiment trajectory tracking (improving/declining/stable)
- Emotion detection and intensity measurement

### 2. Pattern Detection
Automatically detects 50+ critical patterns including:
- **Escalation Patterns**: supervisor requests, threat of legal action, account closure threats
- **Emotional Patterns**: extreme frustration, anger expression, loss of trust
- **Service Issues**: broken promises, unmet expectations, repeat issues
- **Financial Patterns**: billing disputes, unauthorized charges, refund requests
- **Agent Behavior**: questioning competence, empathy, validation

### 3. Vector Database Search
- Semantic search using Sentence Transformers (multi-qa-mpnet-base-dot-v1)
- Persistent ChromaDB storage for efficient retrieval
- Context-aware evidence gathering from similar conversations

### 4. Causal Analysis Engine
- Query expansion using Gemini LLM
- Automatic query categorization
- Evidence-based causal explanations with citations
- Structured output format with recommendations

### 5. Multi-Turn Reasoning
- Session-based conversation tracking
- Evidence locking across conversation turns
- Context preservation for follow-up questions
- Deep-dive analysis on specific transcripts

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Google Gemini API key

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Install dependencies**
```bash
cd Codebase
pip install -r requirements.txt
```

3. **Set up API key**
```python
# Option 1: Environment variable
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Google Colab userdata
from google.colab import userdata
API_KEY = userdata.get('GEMINI_API_KEY')

# Option 3: Direct input (prompted in notebook)
```

## 📊 Data Format

### Input Data Structure

**Conversational_Transcript_Dataset.json**
```json
{
  "transcripts": [
    {
      "transcript_id": "XXXX-XXXX-XXXX-XXXX",
      "time_of_interaction": "2025-10-03 20:22:00",
      "domain": "E-commerce & Retail",
      "intent": "Delivery Investigation",
      "reason_for_call": "Customer reported...",
      "conversation": [
        {
          "speaker": "Agent",
          "text": "Hello, thank you for contacting..."
        },
        {
          "speaker": "Customer",
          "text": "I'm calling about..."
        }
      ]
    }
  ]
}
```

### Output Data Structure

**task1_queries.csv** - Batch Query Results
```csv
Query_Id,Query,Query_Category,System_Output,Remarks
Q001,"Why do customers escalate?","Escalation Analysis","## Causal Explanation...","Evidence: 9 turns..."
```

**Task_2.csv** - Multi-Turn Analysis
```csv
Query_Id,Query_Category,Query,System_Output,Remarks
QS001_A,"Causal Analysis","Why do customers escalate?","## Reasoning...","[transcript_ids]"
QS001_B,"Causal Analysis","Did agent try to de-escalate?","## Reasoning...","[transcript_ids]"
```

## 💻 Usage

### Running the Complete Pipeline

Open `Codebase/app.ipynb` in Jupyter or Google Colab and execute cells sequentially:

1. **Load and Prepare Data**
```python
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Load processed data
turn_df = pd.read_csv("utterances_final.csv")
conv_summary_df = pd.read_csv("conversation_level_summary.csv")
```

2. **Initialize Vector Database**
```python
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1",
    device="cuda"
)

chroma_client = chromadb.PersistentClient(path="chroma_db_turns3")
evidence_collection = chroma_client.get_collection(
    name="evidence_turns",
    embedding_function=embedding_func
)
```

3. **Run Batch Queries**
```python
system = CausalQuerySystem(
    turn_df=turn_df,
    conv_summary_df=conv_summary_df,
    collection=evidence_collection,
    api_key=API_KEY
)

TEST_QUERIES = [
    "Why do customers escalate to supervisors?",
    "What triggers fraud alert investigations?",
    "Why do customers complain about delivery status?"
]

all_results = []
for i, q in enumerate(TEST_QUERIES):
    result = system.query(q)
    # Process and save results
```

4. **Run Multi-Turn Conversations**
```python
task2_system = RobustMultiTurnSystem(system)

TEST_CHAINS = [
    {
        "session_id": "S001",
        "topic": "Escalation",
        "turns": [
            "Why do customers escalate to supervisors?",
            "Did the agent try to de-escalate?",
            "What was the sentiment score when angry?"
        ]
    }
]

for chain in TEST_CHAINS:
    task2_system.start_session(chain['session_id'])
    for i, q in enumerate(chain['turns'], 1):
        output = task2_system.query(q, chain['session_id'], turn_number=i)
```

## 🔍 Key Components

### 1. CausalQuerySystem Class

Main query engine for single-turn causal analysis.

**Key Methods:**
- `_expand_query_intent(user_query)`: Expands queries for better retrieval
- `_get_category(user_query)`: Classifies queries into categories
- `query(user_query, top_k)`: Executes causal analysis
- `_generate_gemini_response()`: Generates structured explanations

### 2. RobustMultiTurnSystem Class

Handles multi-turn conversational reasoning.

**Key Methods:**
- `start_session(session_id)`: Initializes conversation session
- `query(user_query, session_id, turn_number)`: Processes follow-up questions
- Evidence locking ensures consistency across turns

### 3. Pattern Detection

50+ patterns detected including:
```python
patterns = [
    'extreme_frustration', 'escalation_request', 'broken_promise',
    'financial_dispute', 'unmet_expectations', 'loss_of_trust',
    'questioning_competence', 'threat_of_legal_action', ...
]
```

## 📈 Performance Metrics

- **Sentiment Analysis**: Processes ~1000 utterances/minute
- **Vector Indexing**: Handles 5000+ documents efficiently
- **Query Response Time**: 2-5 seconds per query
- **Multi-Turn Latency**: 20-second delay between turns (API rate limiting)

## 🎓 Analysis Capabilities

### Query Categories
1. **Escalation Analysis**: Why customers escalate, agent behavior
2. **Fraud & Security**: Fraud triggers, identity verification issues
3. **Logistics & Delivery**: Delivery failures, tracking problems
4. **Billing & Payments**: Refund requests, billing errors
5. **General Support**: Technical issues, product inquiries

### Output Format

Each analysis includes:
- **Causal Explanation**: 1-sentence summary
- **Key Causal Factors**: 4 evidence-backed factors with citations
- **Recommendations**: 4 actionable recommendations
- **Metadata**: Grounding explanation and confidence level

## 🔧 Configuration

### Adjusting Retrieval Parameters
```python
# Modify top_k for more/fewer evidence points
result = system.query(user_query, top_k=9)  # Default: 9
```

### Changing Embedding Model
```python
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # Lighter model
    device="cpu"  # Use CPU instead of GPU
)
```

### Customizing LLM Prompts
Edit prompts in `CausalQuerySystem._generate_gemini_response()` method.

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Use CPU
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1",
    device="cpu"
)
```

**2. API Rate Limits**
```python
# Solution: Add delays
import time
time.sleep(20)  # Between queries
```

**3. ChromaDB Connection Errors**
```bash
# Solution: Delete and recreate database
rm -rf chroma_db_turns3/
# Re-run indexing cells
```

**4. Missing API Key**
```python
# Solution: Set environment variable
import os
os.environ['GEMINI_API_KEY'] = 'your-key-here'
```

## 📝 Citation Format

All analysis outputs include citations in format:
```
***(Source: XXXX-XXXX-XXXX-XXXX, Turn N)***
```

Where:
- `XXXX-XXXX-XXXX-XXXX`: Transcript ID
- `Turn N`: Specific turn number in conversation

## 🏆 Competition Context

This project was developed for the **ML Hackathon PRAVAAH** competition, focusing on:
- Causal analysis of customer service conversations
- Pattern detection and sentiment analysis
- Multi-turn conversational AI reasoning
- Structured output generation for business insights

## 📄 License

This project is provided as-is for educational and competition purposes.

## 👥 Support

For questions or issues:
1. Review the `ML_HACKATHON_PRAVAAH.pdf` documentation
2. Check the Jupyter notebook comments in `Codebase/app.ipynb`
3. Examine example outputs in `Output/` directory

## 🔗 Dependencies

Key libraries used:
- `pandas`: Data manipulation
- `chromadb`: Vector database
- `sentence-transformers`: Embedding generation
- `google-generativeai`: Gemini LLM integration
- `transformers`: Sentiment analysis models

See `Codebase/requirements.txt` for complete list.
