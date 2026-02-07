# Requirements Document
# ML Hackathon PRAVAAH - Customer Service Conversation Analysis System

## 1. Overview

This document defines the requirements for an AI-powered customer service conversation analysis system that processes conversational transcripts to detect patterns, perform sentiment analysis, and generate causal insights using vector databases and large language models.

## 2. User Stories

### 2.1 Data Processing

**US-1: As a data analyst, I want to process raw conversation transcripts so that I can extract structured utterance-level data**

Acceptance Criteria:
- System loads JSON dataset containing 5,037+ transcripts
- System extracts 84,465+ individual utterances with metadata
- Each utterance includes: transcript_id, turn_no, speaker, text, domain, intent
- Output is stored in structured CSV format

**US-2: As a data scientist, I want to analyze sentiment for each conversation turn so that I can track emotional trajectories**

Acceptance Criteria:
- System uses transformer-based model (RoBERTa) for sentiment analysis
- Sentiment scores range from -1 (negative) to +1 (positive)
- System calculates conversation-level sentiment metrics (avg, min)
- System determines sentiment trajectory (improving/declining/stable)

**US-3: As a business analyst, I want to detect critical patterns in conversations so that I can identify common issues**

Acceptance Criteria:
- System detects 50+ predefined patterns across categories
- Pattern categories include: escalation, emotional, service issues, financial
- System identifies critical turns based on sentiment and patterns
- Pattern detection uses keyword matching with context awareness

### 2.2 Data Storage & Indexing

**US-4: As a developer, I want to store conversation data in a vector database so that I can perform semantic search**

Acceptance Criteria:
- System uses ChromaDB for vector storage
- Embeddings are 768-dimensional using multi-qa-mpnet-base-dot-v1 model
- Each document includes metadata: transcript_id, turn_no, speaker, sentiment, patterns
- Database supports persistent storage to avoid re-indexing

**US-5: As a system administrator, I want aggregated conversation metrics so that I can analyze overall trends**

Acceptance Criteria:
- System generates conversation-level summary with total turns, critical turn count
- Summary includes unique patterns, pattern count, sentiment metrics
- Summary classifies outcomes as 'resolved' or 'escalation'
- Data is stored in CSV format for easy analysis

### 2.3 Query & Analysis

**US-6: As a business user, I want to ask natural language questions about customer behavior so that I can understand root causes**

Acceptance Criteria:
- System accepts natural language queries
- System expands queries using LLM for better retrieval
- System retrieves top-K (default 9) relevant conversation turns
- Query response time is 2-5 seconds

**US-7: As an analyst, I want structured causal explanations so that I can understand why issues occur**

Acceptance Criteria:
- System generates exactly 4 causal factors with source citations
- System provides exactly 4 recommendations with source citations
- Citations include transcript ID and turn number
- Explanations are under 100 words total
- Output includes confidence level (High/Medium/Low)

**US-8: As a researcher, I want to ask follow-up questions so that I can dive deeper into specific cases**

Acceptance Criteria:
- System supports multi-turn conversations with session management
- First query retrieves and locks evidence (transcript IDs)
- Subsequent queries use locked evidence for consistency
- System maintains conversation history across turns
- Maximum 10 turns per session

### 2.4 Batch Processing

**US-9: As a data analyst, I want to process multiple queries in batch so that I can generate reports efficiently**

Acceptance Criteria:
- System accepts list of queries for batch processing
- Output is CSV file with columns: Query_Id, Query, Query_Category, System_Output, Remarks
- System handles API rate limiting with delays
- Batch processing supports parallel execution

### 2.5 Performance & Scalability

**US-10: As a system user, I want fast query responses so that I can work efficiently**

Acceptance Criteria:
- Sentiment analysis processes 1,000+ utterances per minute
- Pattern detection processes 2,000+ utterances per minute
- Query latency (p50) is under 5 seconds
- Query latency (p95) is under 10 seconds
- System supports 10+ concurrent sessions

**US-11: As a developer, I want efficient memory usage so that the system can run on standard hardware**

Acceptance Criteria:
- Memory usage is under 5GB with GPU
- Disk usage for ChromaDB is under 1GB
- System supports lazy loading of transcript data
- Session cleanup releases memory after completion

## 3. Functional Requirements

### 3.1 Sentiment Analysis

**FR-1: Sentiment Scoring**
- Use transformer-based model for sentiment classification
- Generate polarity scores from -1 (negative) to +1 (positive)
- Process text with truncation for long utterances
- Return sentiment intensity measurement

**FR-2: Trajectory Calculation**
- Calculate sentiment trajectory using linear regression
- Classify as "improving" if slope > 0.1
- Classify as "declining" if slope < -0.1
- Classify as "stable" otherwise
- Require minimum 3 sentiment scores for trajectory

### 3.2 Pattern Detection

**FR-3: Multi-Pattern Matching**
- Support 50+ predefined patterns across 10+ categories
- Use case-insensitive keyword matching
- Return unique list of detected patterns per turn
- Support multiple keywords per pattern

**FR-4: Critical Turn Identification**
- Mark turn as critical if sentiment < -0.7
- Mark turn as critical if high-priority patterns detected
- High-priority patterns: escalation_request, extreme_frustration, threat_of_legal_action, financial_dispute
- Track critical turn numbers at conversation level

### 3.3 Query Processing

**FR-5: Query Expansion**
- Use LLM (Gemini) to expand user queries
- Generate hypothetical dialogue turns related to query
- Include relevant keywords (frustrated, supervisor, delivery, fraud)
- Focus on root cause identification

**FR-6: Semantic Retrieval**
- Embed query using same model as documents
- Search ChromaDB for top-K similar turns
- Return results with similarity scores
- Extract full transcript context for retrieved IDs

**FR-7: Query Classification**
- Categorize queries into domains: Escalation Analysis, Fraud & Security, Logistics & Delivery, Billing & Payments, General Support
- Use LLM for automatic classification
- Include category in output metadata

### 3.4 Causal Explanation Generation

**FR-8: Structured Output**
- Generate one-sentence summary of root cause
- Provide exactly 4 causal factors with citations
- Provide exactly 4 recommendations with citations
- Include metadata: grounding explanation, confidence level
- Keep total output under 100 words

**FR-9: Source Citation**
- Format citations as: ***(Source: XXXX-XXXX-XXXX-XXXX, Turn N)***
- Include transcript ID and turn number
- Cite sources for all factors and recommendations
- Ensure citations reference actual retrieved evidence

### 3.5 Multi-Turn Reasoning

**FR-10: Session Management**
- Create unique session ID for each conversation
- Initialize session with empty history and evidence lock
- Track conversation history across turns
- Support session cleanup after completion

**FR-11: Evidence Locking**
- Lock transcript IDs after first query in session
- Use locked evidence for all subsequent queries
- Cache full context to avoid re-retrieval
- Maintain consistency across follow-up questions

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-1: Response Time**
- Single-turn query: 2-5 seconds (p50)
- Single-turn query: < 10 seconds (p95)
- Multi-turn query: < 25 seconds (including API delay)
- Batch processing: > 5 queries per minute

**NFR-2: Throughput**
- Sentiment analysis: 1,000+ utterances/minute
- Pattern detection: 2,000+ utterances/minute
- Vector indexing: 5,000 documents per batch
- Concurrent sessions: 10+ simultaneous users

### 4.2 Scalability

**NFR-3: Data Volume**
- Support 5,000+ conversation transcripts
- Support 100,000+ individual utterances
- Handle 1GB+ vector database size
- Process datasets with 7+ domains, 40+ intents

**NFR-4: Resource Usage**
- Memory: < 5GB with GPU, < 8GB without GPU
- Disk: < 1GB for ChromaDB storage
- GPU: Optional but recommended (CUDA-enabled)
- CPU: Support multi-core parallel processing

### 4.3 Reliability

**NFR-5: Error Handling**
- Handle API rate limiting with automatic retry
- Gracefully handle missing data with defaults
- Validate input queries (non-empty, < 500 chars)
- Validate output structure (required sections, citations)

**NFR-6: Data Integrity**
- Ensure consistent transcript IDs across datasets
- Validate turn numbers are sequential
- Check for duplicate entries during indexing
- Verify embedding dimensions match model output

### 4.4 Usability

**NFR-7: API Design**
- Provide simple query() method for single-turn analysis
- Support intuitive session-based API for multi-turn
- Return structured dictionaries with clear keys
- Include helpful error messages

**NFR-8: Output Format**
- Use markdown formatting for explanations
- Provide clear section headers
- Include metadata for transparency
- Support CSV export for batch results

### 4.5 Maintainability

**NFR-9: Code Organization**
- Separate classes for single-turn and multi-turn systems
- Modular functions for each processing stage
- Clear configuration parameters
- Comprehensive logging

**NFR-10: Documentation**
- Document all public methods and classes
- Provide usage examples
- Include performance benchmarks
- Maintain glossary of terms

### 4.6 Security & Privacy

**NFR-11: Data Protection**
- Anonymize customer identifiers in transcripts
- Use UUID format for transcript IDs
- No storage of sensitive financial information
- Secure API key management (environment variables)

**NFR-12: Access Control**
- Isolate session state per user
- Validate session ID format
- Implement rate limiting (10 requests per 60 seconds)
- Support session timeout and cleanup

## 5. Technical Constraints

### 5.1 Technology Stack

**TC-1: Core Libraries**
- pandas: Data manipulation
- numpy: Numerical computations
- chromadb: Vector database
- sentence-transformers: Embedding generation
- google-generativeai: LLM integration
- transformers: Sentiment analysis

**TC-2: Development Environment**
- Python 3.8+
- Jupyter Notebook / Google Colab
- CUDA-enabled GPU (optional)
- 8GB+ RAM recommended

### 5.2 External Dependencies

**TC-3: APIs**
- Google Gemini API (gemini-2.5-flash model)
- API key required for LLM calls
- Rate limiting: 20 second delay between calls
- Internet connection required

**TC-4: Models**
- Embedding: multi-qa-mpnet-base-dot-v1 (768-dim)
- Sentiment: RoBERTa-based transformer
- Device: CUDA or CPU

### 5.3 Data Format

**TC-5: Input Format**
- JSON: Conversational_Transcript_Dataset.json
- CSV: utterances.csv with required columns
- Text encoding: UTF-8
- Transcript ID format: XXXX-XXXX-XXXX-XXXX (UUID)

**TC-6: Output Format**
- CSV: utterances_final.csv, conversation_level_summary.csv
- ChromaDB: Persistent storage in ./chroma_db_turns3
- Markdown: Structured explanations
- JSON: API responses (dictionaries)

## 6. Acceptance Criteria Summary

### 6.1 Data Processing Pipeline

✓ System loads and processes 5,000+ transcripts
✓ System extracts 80,000+ utterances with metadata
✓ Sentiment analysis achieves 1,000+ utterances/minute
✓ Pattern detection identifies 50+ pattern types
✓ Aggregation generates conversation-level summaries

### 6.2 Storage & Indexing

✓ ChromaDB stores 80,000+ vectors with 768 dimensions
✓ Persistent storage avoids re-indexing
✓ Metadata includes sentiment, patterns, speaker
✓ Disk usage under 1GB

### 6.3 Query & Analysis

✓ Natural language queries return results in 2-5 seconds
✓ Query expansion improves retrieval relevance
✓ Top-K retrieval returns 9 evidence points by default
✓ Causal explanations include 4 factors + 4 recommendations
✓ All outputs include source citations

### 6.4 Multi-Turn Reasoning

✓ Session management supports 10+ concurrent users
✓ Evidence locking ensures consistency across turns
✓ Conversation history preserved for context
✓ Maximum 10 turns per session

### 6.5 Performance

✓ Query latency (p50) under 5 seconds
✓ Query latency (p95) under 10 seconds
✓ Memory usage under 5GB with GPU
✓ Support 10+ concurrent sessions

## 7. Out of Scope

The following features are explicitly out of scope for the current version:

- Real-time streaming data processing
- Web-based user interface or dashboard
- Custom model fine-tuning on domain data
- Distributed vector database deployment
- Multi-language support (English only)
- Audio/video transcript generation
- Integration with CRM systems
- Automated model retraining pipeline
- Advanced visualization tools
- Mobile application support

## 8. Assumptions

- Transcripts are pre-cleaned and formatted
- Customer identifiers are already anonymized
- Internet connection is available for API calls
- Users have basic Python and Jupyter knowledge
- GPU access is optional but recommended
- API keys are managed securely by users
- Dataset fits in memory (< 5GB)
- English language conversations only

## 9. Dependencies

### 9.1 Data Dependencies

- Conversational_Transcript_Dataset.json (5,037 transcripts)
- utterances.csv (84,465 utterances)
- Preprocessed files from google_colab.ipynb

### 9.2 External Service Dependencies

- Google Gemini API (for LLM calls)
- Hugging Face models (for embeddings and sentiment)
- ChromaDB (for vector storage)

### 9.3 Development Dependencies

- Jupyter Notebook environment
- Python package manager (pip)
- Git (for version control)

## 10. Success Metrics

### 10.1 Functional Metrics

- Pattern detection accuracy: 95%+
- Query relevance (user satisfaction): 90%+
- Citation accuracy: 100%
- Output format compliance: 100%

### 10.2 Performance Metrics

- Query latency (p50): < 5 seconds
- Query latency (p95): < 10 seconds
- Throughput: > 5 queries/minute
- Uptime: 99%+ (excluding API downtime)

### 10.3 Quality Metrics

- Code coverage: 80%+
- Documentation completeness: 100%
- Error rate: < 1%
- User-reported bugs: < 5 per month

---

## Document Version

- **Version**: 1.0
- **Date**: February 7, 2026
- **Author**: ML Hackathon PRAVAAH Team
- **Status**: Final
- **Related Documents**: design.md

---

*End of Requirements Document*
