# Deep Research AI Agentic System

## Overview
The Deep Research AI Agentic System is a multi-agent architecture designed to perform deep research by crawling websites, gathering relevant information, and synthesizing it into well-structured answers. The system leverages LangGraph for workflow orchestration, LangChain for integrating tools and managing components, and Tavily for web search and information gathering. It also utilizes the Gemini API for language model capabilities to generate research plans, analyze findings, and draft responses.

This project is designed to work efficiently within the constraints of limited API resources, ensuring high-quality outputs through robust fallback mechanisms, caching, and relevance filtering.

## Features
### Dual-Agent Architecture:
- **Research Agent**: Plans and executes searches, analyzes results, and extracts key facts.
- **Drafting Agent**: Synthesizes findings into a coherent response with an introduction, main content, and conclusion.

### Workflow Orchestration:
- Managed using LangGraph, enabling dynamic decision-making and state transitions.

### Web Crawling:
- Uses Tavily to retrieve information from online sources.

### Language Model Integration:
- Utilizes the Gemini API for generating research plans, analyzing findings, and drafting responses.

### Chunking:
- Splits large documents into smaller, manageable pieces to handle token limits and improve relevance filtering.

### Caching:
- Stores research plans, search results, and draft responses to avoid redundant API calls.

### Fallback Mechanisms:
- Ensures the system provides a response even in cases of partial failures.

### Readable Output:
- Generates answers in paragraph format for better readability and understanding.

## Installation
### Prerequisites
- Python 3.8 or higher
- API keys for:
  - Tavily (for web search)
  - Gemini API (for language model capabilities)

### Steps
1. Clone the repository:
```
git clone https://github.com/prashantsingh5/deep-research-ai.git
cd deep-research-ai
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root and add the following:
   ```
    TAVILY_API_KEY=your_tavily_api_key
    GOOGLE_API_KEY=your_gemini_api_key
   ```

4. Run the system:
```
python Agent.py
```

## Usage
### Running the System
To process a query, run the script and provide a query string. For example:
```
python Agent.py
```

The system will:
- Create a research plan.
- Execute searches using Tavily.
- Analyze findings and extract key facts.
- Draft a response in paragraph format.

### Example Query
```
What are the latest AI technologies in 2025?
```
The system will return a structured response with:
- **Introduction**: A brief overview of the topic.
- **Main Content**: Detailed information in paragraph form.
- **Conclusion**: A summary of the findings.
- **Sources**: Links to the sources used.

## Project Structure
```
deep-research-ai/
│
├── [new2.py](http://_vscodecontentref_/2)                # Main script containing the system implementation
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables (not included in the repo)
```

### Key Components
1. **Research Agent**
   - Creates a research plan with specific search queries.
   - Executes searches using Tavily.
   - Analyzes results to extract key facts, sources, and summaries.

2. **Drafting Agent**
   - Synthesizes findings into a structured response.
   - Ensures the output is readable and well-organized.

3. **Fallback Mechanisms**
   - Handles errors or insufficient data by creating basic drafts or conducting emergency searches.

4. **Caching**
   - Reduces redundant API calls by storing results for reuse.

## Technologies Used
- **LangGraph**: For workflow orchestration and state management.
- **LangChain**: For integrating tools, managing prompts, and handling outputs.
- **Tavily**: For web search and information gathering.
- **Gemini API**: For generating research plans, analyzing findings, and drafting responses.
- **Python**: Core programming language.

## Challenges and Solutions
### 1. API Constraints
- **Challenge**: Limited token usage and rate limits.
- **Solution**: Reduced research rounds, optimized context size, and implemented caching.

### 2. Handling Large Texts
- **Challenge**: Search results often include large amounts of text.
- **Solution**: Used chunking to split documents into smaller pieces for efficient processing.

### 3. Ensuring Relevance
- **Challenge**: Search results may include irrelevant information.
- **Solution**: Implemented relevance filtering to prioritize key terms and focused snippets.

## Future Improvements
- **Semantic Caching**: Enhance the caching mechanism to consider semantic similarity between queries.
- **Quality Evaluation**: Add mechanisms to evaluate the accuracy and reliability of the final response.
- **Advanced Relevance Scoring**: Use machine learning models to improve relevance filtering.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

- **Name**: Your Name
- **Email**: your.email@example.com
- **GitHub**: Your GitHub Profile