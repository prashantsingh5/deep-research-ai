"""
Deep Research AI Agentic System

This system implements a multi-agent architecture for deep research using:
- LangGraph for agent workflow orchestration 
- LangChain for components and tools
- Tavily for web search and information gathering
- Google's Gemini API for language model capabilities
"""

import os
import time
import requests
import urllib3
import warnings
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Disable SSL warnings to prevent terminal clutter
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# Updated import for TavilySearchResults
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
# Consider updating this import if needed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# LangGraph imports
from langgraph.graph import StateGraph, END
# Updated import for MemorySaver
from langgraph.checkpoint.memory import MemorySaver

# Gemini API integration
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import configure

# Define the AgentState class that's used throughout the code
class AgentState(BaseModel):
    """State maintained throughout the agent's execution."""
    query: str
    context: List[str] = Field(default_factory=list)
    research_plan: Optional[Any] = None
    research_findings: List[Any] = Field(default_factory=list)
    urls_explored: List[str] = Field(default_factory=list)
    draft: Optional[Any] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 1  # Reduced to 1 to conserve API usage

# Set default timeout for all requests
requests.adapters.DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 20  # seconds - reduced from 30

# Set user agent for requests
os.environ["USER_AGENT"] = "DeepResearchAI/1.0"
print("USER_AGENT set to:", os.environ["USER_AGENT"])

# Environment setup - consider moving these to environment variables
os.environ["TAVILY_API_KEY"] = "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" 
os.environ["GOOGLE_API_KEY"] = "AIzaSyADtXOCgwP1REFp5gyH6FjIUNH0vxeKjx8" 
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model with retry logic and reduced parameters
def create_llm_with_retry(max_retries=3, retry_delay=2):
    """Create LLM with retry logic in case of API issues"""
    for attempt in range(max_retries):
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.1,  # Reduced temperature for more focused responses
                max_output_tokens=2048,  # Reduced from 4096 to save tokens
                top_p=0.95,
                top_k=40,
                timeout=DEFAULT_TIMEOUT
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"LLM initialization failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                raise

gemini_model = create_llm_with_retry()

# Output schemas for structured responses
class ResearchPlan(BaseModel):
    """Research plan with search queries and strategy."""
    search_queries: List[str] = Field(description="List of search queries to explore the topic")
    research_focus: List[str] = Field(description="Key aspects to focus on during research")
    
class ResearchFindings(BaseModel):
    """Structured research findings from web searches."""
    key_facts: List[str] = Field(description="Important facts discovered during research")
    sources: List[Dict[str, str]] = Field(description="Sources of information with titles and URLs")
    summary: str = Field(description="Summary of the overall findings")
    gaps: List[str] = Field(description="Gaps in the information or areas needing more research")

    @classmethod
    def parse_with_fallback(cls, text_or_dict):
        """Parse the LLM output with fallback mechanism for robustness."""
        try:
            # Check if input is string and try to parse as JSON
            if isinstance(text_or_dict, str):
                try:
                    text_or_dict = json.loads(text_or_dict)
                except json.JSONDecodeError:
                    # Not valid JSON, create basic structure
                    return cls(
                        key_facts=["Information extracted from search results"],
                        sources=[],
                        summary="Search results were found but couldn't be properly structured",
                        gaps=["Complete structured analysis"]
                    )
            
            if isinstance(text_or_dict, dict):
                # Try to extract the fields we need
                sources_data = []
                if "sources" in text_or_dict:
                    raw_sources = text_or_dict["sources"]
                    if isinstance(raw_sources, list):
                        for src in raw_sources:
                            if isinstance(src, str):
                                sources_data.append({"title": src, "url": src})
                            elif isinstance(src, dict) and "url" in src:
                                sources_data.append({"title": src.get("title", src["url"]), "url": src["url"]})
                
                # Handle various field naming patterns
                key_facts = text_or_dict.get("key_facts", text_or_dict.get("keyFacts", []))
                if isinstance(key_facts, str):
                    # If it's a string, try to split it into list items
                    key_facts = [fact.strip() for fact in key_facts.split('\n') if fact.strip()]
                elif isinstance(key_facts, list) and all(isinstance(item, dict) for item in key_facts):
                    key_facts = [item.get("fact", str(item)) for item in key_facts]
                
                summary = text_or_dict.get("summary", text_or_dict.get("overallSummary", "No summary available"))
                gaps = text_or_dict.get("gaps", text_or_dict.get("gapsInInformation", []))
                
                if isinstance(gaps, str):
                    gaps = [gap.strip() for gap in gaps.split('\n') if gap.strip()]
                
                return cls(
                    key_facts=key_facts if isinstance(key_facts, list) else [],
                    sources=sources_data,
                    summary=summary if isinstance(summary, str) else "No summary available",
                    gaps=gaps if isinstance(gaps, list) else []
                )
            return cls(
                key_facts=["Information gathered from search results"],
                sources=[],
                summary="Search results were found but couldn't be properly structured",
                gaps=["Complete data extraction"]
            )
        except Exception as e:
            print(f"Fallback parsing for ResearchFindings: {str(e)}")
            return cls(
                key_facts=["Information found but processing encountered errors"],
                sources=[],
                summary=f"Information was collected but encountered formatting issues: {str(e)}",
                gaps=["Complete analysis due to processing error"]
            )


class DraftResponse(BaseModel):
    """Draft response to the user query."""
    introduction: str = Field(description="Brief introduction to the topic")
    main_content: str = Field(description="Comprehensive answer to the query")
    conclusion: str = Field(description="Concluding thoughts and summary")
    sources: List[Dict[str, str]] = Field(description="Sources used in the response")

    @classmethod
    def parse_with_fallback(cls, text_or_dict):
        """Parse the LLM output with fallback mechanism for robustness."""
        try:
            # Check if input is string and try to parse as JSON
            if isinstance(text_or_dict, str):
                try:
                    text_or_dict = json.loads(text_or_dict)
                except json.JSONDecodeError:
                    # Not valid JSON, check if it's structured text
                    sections = text_or_dict.split("\n\n")
                    intro = ""
                    main = ""
                    conclusion = ""
                    
                    for section in sections:
                        if "introduction" in section.lower() or "overview" in section.lower():
                            intro = section
                        elif "conclusion" in section.lower() or "summary" in section.lower():
                            conclusion = section
                        else:
                            main += section + "\n\n"
                    
                    return cls(
                        introduction=intro if intro else "Introduction to the topic based on research",
                        main_content=main if main else text_or_dict,
                        conclusion=conclusion if conclusion else "End of research findings",
                        sources=[]
                    )
            
            if isinstance(text_or_dict, dict):
                # Check if there's a nested draft structure
                if "draft" in text_or_dict and isinstance(text_or_dict["draft"], dict):
                    draft_dict = text_or_dict["draft"]
                    
                    # Extract sources/citations
                    sources_data = []
                    if "sources" in text_or_dict:
                        raw_sources = text_or_dict["sources"]
                    elif "citations" in draft_dict:
                        raw_sources = draft_dict["citations"]
                    elif "sources" in draft_dict:
                        raw_sources = draft_dict["sources"]
                    else:
                        raw_sources = []
                        
                    if isinstance(raw_sources, list):
                        for src in raw_sources:
                            if isinstance(src, str):
                                sources_data.append({"title": src, "url": src})
                            elif isinstance(src, dict) and "url" in src:
                                sources_data.append({"title": src.get("title", src["url"]), "url": src["url"]})
                    
                    # Extract main content - handle both string and structured formats
                    main_content = draft_dict.get("main_content", "")
                    if isinstance(main_content, list):
                        # If it's structured content, convert to string
                        content_parts = []
                        for section in main_content:
                            if isinstance(section, dict):
                                # Add heading if available
                                if "heading" in section:
                                    content_parts.append(f"## {section['heading']}")
                                
                                # Add paragraphs
                                if "paragraphs" in section and isinstance(section["paragraphs"], list):
                                    content_parts.extend(section["paragraphs"])
                        
                        main_content = "\n\n".join(content_parts)
                    
                    return cls(
                        introduction=draft_dict.get("introduction", "Introduction not available"),
                        main_content=main_content if isinstance(main_content, str) else "Content not available",
                        conclusion=draft_dict.get("conclusion", "Conclusion not available"),
                        sources=sources_data
                    )
                
                # Handle case where response is directly in the top level
                if "response" in text_or_dict and isinstance(text_or_dict["response"], dict):
                    response_dict = text_or_dict["response"]
                    
                    # Extract main content from response structure
                    main_content = response_dict.get("main_content", "")
                    if isinstance(main_content, list):
                        content_parts = []
                        for section in main_content:
                            if isinstance(section, dict):
                                if "heading" in section:
                                    content_parts.append(f"## {section['heading']}")
                                
                                if "paragraphs" in section and isinstance(section["paragraphs"], list):
                                    content_parts.extend(section["paragraphs"])
                        
                        main_content = "\n\n".join(content_parts)
                    
                    # Handle sources
                    sources_data = []
                    if "sources" in text_or_dict:
                        raw_sources = text_or_dict["sources"]
                        if isinstance(raw_sources, list):
                            for src in raw_sources:
                                if isinstance(src, dict) and "url" in src:
                                    sources_data.append({"title": src.get("title", src["url"]), "url": src["url"]})
                    
                    return cls(
                        introduction=response_dict.get("introduction", "Introduction not available"),
                        main_content=main_content if isinstance(main_content, str) else "Content not available",
                        conclusion=response_dict.get("conclusion", "Conclusion not available"),
                        sources=sources_data
                    )
                
                # Simple direct mapping case
                intro = text_or_dict.get("introduction", "")
                content = text_or_dict.get("main_content", "")
                conclusion = text_or_dict.get("conclusion", "")
                sources = text_or_dict.get("sources", [])
                
                if not intro and not content and not conclusion:
                    # Nothing found in expected fields, create a placeholder
                    return cls(
                        introduction="Information about this topic is limited due to processing errors.",
                        main_content="The research system encountered validation errors while processing the data. However, some information was gathered.",
                        conclusion="Please try another search query for better results.",
                        sources=[]
                    )
                
                return cls(
                    introduction=intro if isinstance(intro, str) else "Introduction not available",
                    main_content=content if isinstance(content, str) else "Content not available",
                    conclusion=conclusion if isinstance(conclusion, str) else "Conclusion not available",
                    sources=sources if isinstance(sources, list) else []
                )
            
            # Default fallback
            return cls(
                introduction="Information gathered had formatting issues",
                main_content="The research system encountered errors while processing the data",
                conclusion="Please try another search or rephrase your query",
                sources=[]
            )
        except Exception as e:
            print(f"Fallback parsing for DraftResponse: {str(e)}")
            return cls(
                introduction="Error occurred during research",
                main_content=f"The system encountered an error while processing the research data: {str(e)}",
                conclusion="Please try another query or contact support if this issue persists",
                sources=[]
            )

# Tools initialization with error handling and caching
try:
    # Initialize Tavily tool
    tavily_search_tool = TavilySearchResults(k=3)  # reduced from 5
    print("Tavily search tool initialized successfully")
except Exception as e:
    print(f"Failed to initialize Tavily search tool: {str(e)}")
    print("Using a dummy search tool as fallback")
    
    # Create a dummy search tool as fallback
    class DummySearchTool:
        def invoke(self, query_dict):
            return [
                {
                    "url": "https://example.com/dummy",
                    "content": f"This is dummy content for the query: {query_dict.get('query', '')}",
                    "title": "Dummy Search Result"
                }
            ]
    
    tavily_search_tool = DummySearchTool()

# Create a simple caching mechanism to avoid redundant API calls
search_cache = {}

# =====================
# Research Agent
# =====================

def create_research_plan(state: AgentState) -> AgentState:
    """Create a research plan based on the user query."""
    try:
        # Check cache first
        cache_key = f"plan_{state.query}"
        if cache_key in search_cache:
            print("Using cached research plan")
            return AgentState(
                query=state.query,
                research_plan=search_cache[cache_key],
                messages=state.messages + [{"role": "system", "content": "Retrieved cached research plan"}],
                current_round=state.current_round,
                max_rounds=state.max_rounds
            )
        
        research_planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research planning expert. Create a focused research plan for the topic.
            You must provide exactly 2-3 specific search queries that will yield highly relevant information.
            
            Your output must be a valid JSON object with these fields:
            - search_queries: List of 2-3 specific search queries 
            - research_focus: List of 2-3 key areas to focus on"""),
            ("human", "{query}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        chain = research_planner_prompt | gemini_model | parser
        
        research_plan = chain.invoke({"query": state.query})
        print(f"Research plan created with {len(research_plan.search_queries)} search queries")
        
        # Store in cache
        search_cache[cache_key] = research_plan
        
        return AgentState(
            query=state.query,
            research_plan=research_plan,
            messages=state.messages + [{"role": "system", "content": f"Created research plan with {len(research_plan.search_queries)} search queries"}],
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )
    except Exception as e:
        print(f"Error creating research plan: {str(e)}")
        # Create a basic fallback plan
        fallback_plan = ResearchPlan(
            search_queries=[state.query, f"latest information on {state.query}"],
            research_focus=["General information", "Latest developments"]
        )
        print(f"Using fallback research plan with {len(fallback_plan.search_queries)} queries")
        return AgentState(
            query=state.query,
            research_plan=fallback_plan,
            messages=state.messages + [{"role": "system", "content": f"Error creating research plan: {str(e)}. Using fallback plan."}],
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )

def execute_tavily_search(state: AgentState) -> AgentState:
    """Execute searches using Tavily search engine with improved relevance filtering."""
    findings = state.research_findings.copy() if state.research_findings else []
    urls_explored = state.urls_explored.copy() if state.urls_explored else []
    context = state.context.copy() if state.context else []
    
    # Use queries from the research plan - limit to 2 most relevant
    for i, query in enumerate(state.research_plan.search_queries[:2]):
        cache_key = f"search_{query}"
        search_results = []
        
        # Check cache first
        if cache_key in search_cache:
            search_results = search_cache[cache_key]
            print(f"Using cached search results for: {query}")
        else:
            try:
                print(f"Searching for: {query}")
                search_results = tavily_search_tool.invoke({"query": query, "max_results": 3})
                # Store in cache
                search_cache[cache_key] = search_results
            except Exception as e:
                print(f"Search failed for query '{query}': {str(e)}")
                context.append(f"Search failed for query '{query}': {str(e)}")
                continue
        
        # Process search results with improved relevance filtering
        for result in search_results:
            if result["url"] not in urls_explored:
                # Filter for relevance - only include if title or content contains key terms
                query_terms = set(query.lower().split())
                content_text = (result.get("content", "") + result.get("title", "")).lower()
                
                # Check relevance by counting query terms in content
                relevant_terms = [term for term in query_terms if term in content_text]
                if len(relevant_terms) >= 1:  # at least one query term must be present
                    urls_explored.append(result["url"])
                    
                    # Extract the most relevant snippet (up to 500 chars)
                    if "content" in result and result["content"]:
                        # Find the most relevant part of the content
                        best_snippet = result["content"]
                        if len(best_snippet) > 500:
                            # Try to extract a more focused snippet around keyword
                            for term in relevant_terms:
                                if term in best_snippet.lower():
                                    pos = best_snippet.lower().find(term)
                                    start = max(0, pos - 200)
                                    end = min(len(best_snippet), pos + 300)
                                    best_snippet = best_snippet[start:end]
                                    break
                        
                        context.append(f"From '{result.get('title', 'Untitled')}' ({result['url']}): {best_snippet}")
                    
                    # Only load detailed content for highly relevant results
                    if len(relevant_terms) >= 2 and len(context) < 8:
                        try:
                            # Add timeout to prevent hanging on slow websites
                            loader = WebBaseLoader(
                                result["url"],
                                requests_kwargs={"timeout": DEFAULT_TIMEOUT},
                                verify_ssl=False  # Skip SSL verification for faster loading
                            )
                            docs = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                            splits = text_splitter.split_documents(docs)
                            
                            print(f"Loaded {len(splits)} text chunks from {result['url']}")
                            
                            # Only include the most relevant chunk
                            if splits:
                                most_relevant_chunk = splits[0]
                                for chunk in splits[:3]:
                                    chunk_text = chunk.page_content.lower()
                                    chunk_relevance = sum(term in chunk_text for term in query_terms)
                                    if chunk_relevance > sum(term in most_relevant_chunk.page_content.lower() for term in query_terms):
                                        most_relevant_chunk = chunk
                                
                                context.append(f"Detailed from {result['url']}: {most_relevant_chunk.page_content}")
                            
                        except Exception as e:
                            print(f"Failed to load detailed content from {result['url']}: {str(e)}")
    
    # If we have too little context, add a placeholder
    if len(context) < 2:
        context.append("Limited search results were found. Please try with different search terms.")
    
    # Create research findings summary with more focused prompting
    try:
        research_findings_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research analyst. Analyze the provided information from web searches.
            Focus only on extracting factual information directly related to the query.
            Return a concise and focused analysis with key facts, sources, summary and gaps.
            
            Format your response as JSON with these fields:
            - key_facts: array of 3-5 most important facts discovered
            - sources: array of objects with title and url fields
            - summary: brief factual summary in 2-3 sentences
            - gaps: array of 1-2 key areas needing more information"""),
            ("human", "Query: {query}\nFocus areas: {research_focus}\nData: {context}")
        ])
        
        chain = research_findings_prompt | gemini_model
        raw_research_finding = chain.invoke({
            "query": state.query,
            "research_focus": ", ".join(state.research_plan.research_focus[:2]),  # Limit to top 2 focus areas
            "context": "\n\n".join(context[:8])  # Limit context to reduce token usage
        })
        
        # Use our custom parse method instead of the standard parser
        research_finding = ResearchFindings.parse_with_fallback(raw_research_finding)
        findings.append(research_finding)
        
        print(f"Created research finding with {len(research_finding.key_facts)} key facts")
    except Exception as e:
        print(f"Error creating research findings: {str(e)}")
        # Create fallback findings
        fallback_finding = ResearchFindings(
            key_facts=["Information gathered from web search"],
            sources=[{"title": url, "url": url} for url in urls_explored[:3]],
            summary=f"Research on '{state.query}' found some information from {len(urls_explored)} sources.",
            gaps=["Detailed analysis"]
        )
        findings.append(fallback_finding)
        print("Created fallback research finding")
    
    return AgentState(
        query=state.query,
        context=context,
        research_plan=state.research_plan,
        research_findings=findings,
        urls_explored=urls_explored,
        messages=state.messages + [{"role": "system", "content": f"Completed research with {len(urls_explored)} sources analyzed"}],
        current_round=state.current_round + 1,
        max_rounds=state.max_rounds
    )

def should_continue_research(state: AgentState) -> str:
    """Simplified decision logic for continuing research to save API calls."""
    # Simplified decision logic to conserve API usage
    if state.current_round >= state.max_rounds:
        print(f"Reached maximum research rounds ({state.max_rounds}). Moving to draft.")
        return "draft_answer"
    
    # With limited API calls, almost always move to draft answer after first round
    print("Moving to draft answer to conserve API usage")
    return "draft_answer"

def refine_research(state: AgentState) -> AgentState:
    """Refine the research plan based on gaps identified."""
    # This function is mostly unused with simplified should_continue_research
    try:
        # Get gaps from the last research findings
        gaps = state.research_findings[-1].gaps if state.research_findings else ["Complete information"]
        
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research strategy expert. Based on the research conducted and
            identified gaps, refine the research plan with new search queries.
            
            Be extremely concise and focused. Your output must be a valid JSON with:
            - search_queries: array of 1-2 specific search queries for the gaps
            - research_focus: array of 1-2 key areas to focus on"""),
            ("human", """
            Query: {query}
            Initial focus: {initial_focus}
            Gaps identified: {gaps}
            """)
        ])
        
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        chain = refinement_prompt | gemini_model | parser
        
        refined_plan = chain.invoke({
            "query": state.query,
            "initial_focus": ", ".join(state.research_plan.research_focus[:2]),
            "gaps": ", ".join(gaps[:2])
        })
        
        print(f"Refined research plan with {len(refined_plan.search_queries)} new search queries")
        
        return AgentState(
            query=state.query,
            context=state.context,
            research_plan=refined_plan,
            research_findings=state.research_findings,
            urls_explored=state.urls_explored,
            messages=state.messages + [{"role": "system", "content": f"Refined research plan to address {len(gaps)} gaps"}],
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )
    except Exception as e:
        print(f"Error refining research plan: {str(e)}")
        # Keep the existing plan but add one new query based on gaps
        gaps = state.research_findings[-1].gaps if state.research_findings else ["Complete information"]
        new_queries = [f"{state.query} {gaps[0]}"] if gaps else [state.query]
        refined_plan = ResearchPlan(
            search_queries=new_queries,
            research_focus=state.research_plan.research_focus[:2] if state.research_plan else ["General information"]
        )
        
        print(f"Created basic refined plan with {len(refined_plan.search_queries)} queries")
        
        return AgentState(
            query=state.query,
            context=state.context,
            research_plan=refined_plan,
            research_findings=state.research_findings,
            urls_explored=state.urls_explored,
            messages=state.messages + [{"role": "system", "content": f"Error refining plan: {str(e)}. Using basic refinement."}],
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )

# =====================
# Draft Agent
# =====================

def draft_answer(state: AgentState) -> AgentState:
    """Draft a comprehensive answer based on research findings."""
    # Collect all findings
    all_key_facts = []
    all_sources = []
    summaries = []
    
    if state.research_findings:
        print(f"Creating draft from {len(state.research_findings)} research findings")
        for finding in state.research_findings:
            all_key_facts.extend(finding.key_facts[:5])  # Limit key facts to reduce token usage
            all_sources.extend(finding.sources[:3])      # Limit sources
            summaries.append(finding.summary)
    else:
        print("No research findings available, creating basic draft")
    
    # Create a consolidated context for drafting - more focused to save tokens
    consolidated_context = "\n\n".join([
        f"Query: {state.query}",
        f"Focus areas: {', '.join(state.research_plan.research_focus[:2] if state.research_plan else ['General information'])}",
        f"Key facts:\n" + "\n".join([f"- {fact}" for fact in all_key_facts[:8]]),  # Limit to top 8 facts
        f"Summary: " + " ".join(summaries[:2])  # Limit to first 2 summaries
    ])
    
    # Check if we already have a cached draft for this query
    cache_key = f"draft_{state.query}"
    if cache_key in search_cache:
        print("Using cached draft")
        draft = search_cache[cache_key]
        # Ensure sources are included
        if not draft.sources:
            draft.sources = all_sources
            
        return AgentState(
            query=state.query,
            context=state.context,
            research_plan=state.research_plan,
            research_findings=state.research_findings,
            urls_explored=state.urls_explored,
            draft=draft,
            messages=state.messages + [{"role": "system", "content": "Retrieved cached draft answer"}],
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )
    
    # Draft the answer with more specific instructions to save tokens
    try:
        drafting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content creator. Create a well-structured, easy-to-understand answer 
            using only the provided research information. Focus on accuracy, relevance and readability.
            
            Write in a clear, engaging style with:
            - A brief introduction (2-3 sentences)
            - Main content organized in coherent paragraphs (not bullet points)
            - Each paragraph should focus on one aspect of the topic
            - A short conclusion (1-2 sentences)
            
            IMPORTANT: Format your main content as flowing paragraphs rather than bullet points.
            Connect ideas logically between paragraphs for a smooth reading experience.
            
            Format as JSON with these fields:
            - introduction: brief intro
            - main_content: detailed answer in paragraph form 
            - conclusion: brief conclusion
            - sources: keep empty array, will be filled separately"""),
            ("human", "{context}")
        ])
        
        chain = drafting_prompt | gemini_model
        raw_draft = chain.invoke({"context": consolidated_context})
        
        # Use our custom parse method
        draft = DraftResponse.parse_with_fallback(raw_draft)
        
        # Always ensure sources are properly included
        draft.sources = all_sources
            
        print(f"Draft created: intro ({len(draft.introduction)} chars), content ({len(draft.main_content)} chars)")
        
        # Cache the draft
        search_cache[cache_key] = draft
        
    except Exception as e:
        print(f"Error creating draft: {str(e)}")
        # Create a basic fallback draft in paragraph format
        draft = DraftResponse(
            introduction=f"Based on research about '{state.query}', we've gathered the following information.",
            main_content="".join([
                "Our research has revealed several important points about this topic. ",
                " ".join([fact for fact in all_key_facts[:5]]) + " ",
                "These insights provide context for understanding the subject matter. ",
                " ".join(summaries[:1]) + " ",
                "The information suggests that this is a topic with various dimensions worth exploring further."
            ]),
            conclusion=f"In conclusion, these findings represent an informative overview of '{state.query}', though further research could reveal additional insights.",
            sources=all_sources
        )
        print("Created fallback draft in paragraph format")
    
    return AgentState(
        query=state.query,
        context=state.context,
        research_plan=state.research_plan,
        research_findings=state.research_findings,
        urls_explored=state.urls_explored,
        draft=draft,
        messages=state.messages + [{"role": "system", "content": "Generated draft answer"}],
        current_round=state.current_round,
        max_rounds=state.max_rounds
    )

def finalize_answer(state: AgentState) -> AgentState:
    """Finalize the answer with minimal processing to save API calls."""
    
    if not state.draft:
        print("No draft available, creating minimal draft")
        # Create minimal draft if none exists
        draft = DraftResponse(
            introduction=f"Based on our research into '{state.query}', we've compiled the following information.",
            main_content="The system has gathered information from multiple sources, analyzing relevant content about the topic. While processing the data, we focused on extracting accurate and helpful details that address the core question. The available information suggests this is a topic with multiple aspects worth understanding.",
            conclusion="We hope this information proves helpful in understanding the topic. For more specific details, consider refining your query.",
            sources=[{"title": url, "url": url} for url in state.urls_explored[:3]] if state.urls_explored else []
        )
        state = AgentState(
            query=state.query,
            context=state.context,
            research_plan=state.research_plan,
            research_findings=state.research_findings,
            urls_explored=state.urls_explored,
            draft=draft,
            messages=state.messages,
            current_round=state.current_round,
            max_rounds=state.max_rounds
        )
    
    # Skip verification to save API calls - just do minimal formatting improvements
    try:
        print("Performing minimal draft refinement")
        
        # Check for cached final version
        cache_key = f"final_{state.query}"
        if cache_key in search_cache:
            print("Using cached final draft")
            return AgentState(
                query=state.query,
                context=state.context,
                research_plan=state.research_plan,
                research_findings=state.research_findings,
                urls_explored=state.urls_explored,
                draft=search_cache[cache_key],
                messages=state.messages + [{"role": "system", "content": "Retrieved cached final answer"}],
                current_round=state.current_round,
                max_rounds=state.max_rounds
            )
            
        # Light formatting fixes only
        intro = state.draft.introduction
        if not intro.endswith('.'):
            intro += '.'
            
        content = state.draft.main_content
        if not content.strip():
            # Create paragraph content instead of bullet points
            all_facts = []
            for finding in state.research_findings:
                if hasattr(finding, 'key_facts'):
                    all_facts.extend(finding.key_facts)
            
            if all_facts:
                content = "Our research has revealed important information about this topic. "
                content += " ".join(all_facts[:5]) + " "
                content += "These findings represent the most relevant information we could gather from credible sources. "
                content += "The research process involved analyzing multiple perspectives to provide a comprehensive overview."
            else:
                content = "Our research system gathered information about your query, though the results were limited. The available data suggests this is a topic that merits attention, though specific details were difficult to extract from the available sources. We've compiled what information we could find to provide some context on the subject matter."
                      
        conclusion = state.draft.conclusion
        if not conclusion.endswith('.'):
            conclusion += '.'
            
        # Create lightly refined draft
        refined_draft = DraftResponse(
            introduction=intro,
            main_content=content,
            conclusion=conclusion,
            sources=state.draft.sources
        )
        
        # Cache the final version
        search_cache[cache_key] = refined_draft
        print("Minimal draft refinement complete")
        
    except Exception as e:
        print(f"Error in minimal formatting: {str(e)}")
        refined_draft = state.draft
    
    return AgentState(
        query=state.query,
        context=state.context,
        research_plan=state.research_plan,
        research_findings=state.research_findings,
        urls_explored=state.urls_explored,
        draft=refined_draft,
        messages=state.messages + [
            {"role": "system", "content": "Completed minimal draft refinement"},
            {"role": "system", "content": "Research and drafting process complete"}
        ],
        current_round=state.current_round,
        max_rounds=state.max_rounds
    )

# =====================
# LangGraph Implementation
# =====================

def create_research_graph():
    """Create the research workflow graph using LangGraph."""
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("create_research_plan", create_research_plan)
    workflow.add_node("execute_tavily_search", execute_tavily_search)
    workflow.add_node("refine_research", refine_research)
    workflow.add_node("draft_answer", draft_answer)
    workflow.add_node("finalize_answer", finalize_answer)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "execute_tavily_search",
        should_continue_research,
        {
            "continue_research": "refine_research",
            "draft_answer": "draft_answer"
        }
    )
    
    # Add standard edges
    workflow.add_edge("create_research_plan", "execute_tavily_search")
    workflow.add_edge("refine_research", "execute_tavily_search")
    workflow.add_edge("draft_answer", "finalize_answer")
    workflow.add_edge("finalize_answer", END)
    
    # Set the entrypoint
    workflow.set_entry_point("create_research_plan")
    
    # Compile the graph
    return workflow.compile()

# =====================
# Main System Interface
# =====================

class DeepResearchSystem:
    """Main interface for the Deep Research AI Agentic System."""
    
    def __init__(self):
        print("Initializing Deep Research System...")
        self.graph = create_research_graph()
        self.last_final_state = None
        print("System initialized successfully")
    
    def process_query(self, query: str, max_research_rounds: int = 1) -> str:
        """Process a research query through the agentic system."""
        print(f"\nStarting research on: {query}")
        
        # Initialize the state
        initial_state = AgentState(
            query=query,
            context=[],
            research_findings=[],
            urls_explored=[],
            messages=[],
            max_rounds=max_research_rounds
        )
        
        # Track the final state
        final_state = None
        
        try:
            # Add a global timeout for the whole process
            start_time = time.time()
            max_execution_time = 120  # reduced from 180 to 120 seconds max
            
            # Check cache first
            cache_key = f"final_result_{query}"
            if cache_key in search_cache:
                print("Using cached final results")
                return search_cache[cache_key]
            
            # Execute the graph with timeout handling
            print("Starting research workflow execution...")
            events = []
            for event in self.graph.stream(initial_state):
                # Check for timeout
                if time.time() - start_time > max_execution_time:
                    print(f"Execution timeout after {max_execution_time} seconds")
                    break
                
                # Store events for later analysis
                events.append(event)
                
                # Process event to track state
                if isinstance(event, dict):
                    if "state" in event:
                        final_state = event["state"]
                    elif "data" in event and "state" in event["data"]:
                        final_state = event["data"]["state"]
                elif hasattr(event, "state"):
                    final_state = event.state
            
            # If we still don't have a final state, use the last event
            if not final_state:
                print("Warning: Could not determine final state from events stream")
                # Process last event or create fallback
                final_state = self._create_fallback_state(events, initial_state)
        except Exception as e:
            print(f"Error during graph execution: {str(e)}")
            # Create fallback state
            final_state = self._create_fallback_state([], initial_state)
        
        # Store the final state for statistics
        self.last_final_state = final_state

        # Format the final answer from the draft
        if final_state and hasattr(final_state, "draft") and final_state.draft:
            final_answer = f"""
# {final_state.query}

## Introduction
{final_state.draft.introduction}

## Main Content
{final_state.draft.main_content}

## Conclusion
{final_state.draft.conclusion}

## Sources
"""
            for i, source in enumerate(final_state.draft.sources, 1):
                if isinstance(source, dict) and "title" in source and "url" in source:
                    final_answer += f"{i}. [{source['title']}]({source['url']})\n"
                elif isinstance(source, dict) and "url" in source:
                    final_answer += f"{i}. [Source {i}]({source['url']})\n"
                elif isinstance(source, str):
                    final_answer += f"{i}. {source}\n"
            
            # Cache the final result
            search_cache[cache_key] = final_answer
            return final_answer
        else:
            # Create a basic response if everything fails
            basic_response = f"""
# {query}

## Introduction
The research system encountered issues while processing your query.

## Main Content
While the AI agent searched for information on "{query}", it could not generate a complete response 
due to API rate limits or processing issues.

## Conclusion
Please try again with a more specific query, or try later when API resources are available.

## Sources
No sources could be properly processed.
"""
            # Cache even this fallback result
            search_cache[cache_key] = basic_response
            return basic_response
    
    def _create_fallback_state(self, events, initial_state):
        """Helper method to create fallback state when normal execution fails."""
        print("Creating fallback state")
        try:
            # Try to extract useful information from events if available
            research_findings = []
            urls_explored = []
            
            for event in events:
                if isinstance(event, dict):
                    if "state" in event and hasattr(event["state"], "research_findings"):
                        research_findings = event["state"].research_findings
                        urls_explored = getattr(event["state"], "urls_explored", [])
                        break
                    elif "data" in event and "state" in event["data"] and hasattr(event["data"]["state"], "research_findings"):
                        research_findings = event["data"]["state"].research_findings
                        urls_explored = getattr(event["data"]["state"], "urls_explored", [])
                        break
                elif hasattr(event, "state") and hasattr(event.state, "research_findings"):
                    research_findings = event.state.research_findings
                    urls_explored = getattr(event.state, "urls_explored", [])
                    break
            
            # If we have some research findings, create a basic draft
            if research_findings:
                print("Creating fallback draft from partial research")
                all_key_facts = []
                all_sources = []
                
                for finding in research_findings:
                    if hasattr(finding, "key_facts"):
                        all_key_facts.extend(finding.key_facts)
                    if hasattr(finding, "sources"):
                        all_sources.extend(finding.sources)
                
                # Create paragraph-formatted content instead of bullet points
                main_content = ""
                if all_key_facts:
                    main_content = "Our research has revealed important information about this topic. "
                    main_content += " ".join(all_key_facts[:5]) + " "
                    main_content += "These findings represent the most relevant information we could gather from credible sources. "
                    main_content += "The research process involved analyzing multiple perspectives to provide a comprehensive overview."
                else:
                    main_content = "Our research system gathered information about your query, though the results were limited. The available data suggests this is a topic that merits attention, though specific details were difficult to extract from the available sources. We've compiled what information we could find to provide some context on the subject matter."
                
                draft = DraftResponse(
                    introduction=f"This report presents our research findings on '{initial_state.query}'.",
                    main_content=main_content,
                    conclusion="While the system faced some technical limitations, we hope this information proves useful in understanding the topic.",
                    sources=all_sources
                )
                
                return AgentState(
                    query=initial_state.query,
                    research_findings=research_findings,
                    urls_explored=urls_explored,
                    draft=draft,
                    messages=initial_state.messages + [{"role": "system", "content": "Created fallback draft"}]
                )
            
            # If no research yet, try to run search directly
            print("Running emergency search")
            try:
                # Basic search
                search_results = tavily_search_tool.invoke({"query": initial_state.query, "max_results": 3})
                
                # Create minimal findings and draft
                sources = []
                key_facts = ["We found some information related to your query."]

                for result in search_results:
                    sources.append({"title": result.get("title", "Untitled"), "url": result["url"]})
                    if "content" in result and result["content"]:
                        key_facts.append(result["content"][:200] + "...")

                finding = ResearchFindings(
                    key_facts=key_facts,
                    sources=sources,
                    summary=f"Basic information about '{initial_state.query}' was retrieved.",
                    gaps=["Detailed analysis"]
                )

                # Create paragraph-formatted content
                main_content = f"Our emergency search for information about '{initial_state.query}' yielded some results. "
                main_content += " ".join(key_facts) + " "
                main_content += "While this represents only basic information, it provides a starting point for understanding the topic. "
                main_content += "The search focused on finding the most relevant and recent information available."

                draft = DraftResponse(
                    introduction=f"Here's what we found about '{initial_state.query}':",
                    main_content=main_content,
                    conclusion="This represents the basic information we could gather through our search process.",
                    sources=sources
                )
                
                return AgentState(
                    query=initial_state.query,
                    research_findings=[finding],
                    urls_explored=[s["url"] for s in sources],
                    draft=draft,
                    messages=initial_state.messages + [{"role": "system", "content": "Created emergency draft"}]
                )
            except Exception as e:
                print(f"Emergency search failed: {str(e)}")
        
        except Exception as e:
            print(f"Error in fallback state creation: {str(e)}")
        
        # Ultimate fallback - empty draft
        print("Creating empty draft")
        draft = DraftResponse(
            introduction=f"Results for query: {initial_state.query}",
            main_content="The research system encountered technical limitations with the API.",
            conclusion="Please try again later or with a different query.",
            sources=[]
        )
        
        return AgentState(
            query=initial_state.query,
            draft=draft,
            messages=initial_state.messages + [{"role": "system", "content": "Created empty draft due to errors"}]
        )
            
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get statistics about the last research process."""
        if not self.last_final_state:
            return {"status": "No research conducted yet"}
        
        try:
            last_state = self.last_final_state
            stats = {
                "query": last_state.query,
                "research_rounds": getattr(last_state, "current_round", 0),
                "urls_explored": len(getattr(last_state, "urls_explored", [])),
                "search_queries_used": len(last_state.research_plan.search_queries) if hasattr(last_state, "research_plan") and last_state.research_plan else 0,
                "key_facts_discovered": sum(len(finding.key_facts) for finding in getattr(last_state, "research_findings", []) if hasattr(finding, "key_facts")),
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cache_items": len(search_cache)
            }
            print("Research statistics calculated successfully")
            return stats
        except Exception as e:
            print(f"Error generating stats: {str(e)}")
            return {
                "query": getattr(self.last_final_state, "query", "Unknown"),
                "error": f"Error generating statistics: {str(e)}",
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

# Usage example
if __name__ == "__main__":
    system = DeepResearchSystem()
    
    # Test the system with a query about latest AI technology
    try:
        result = system.process_query("What are the latest AI technologies in 2025?", max_research_rounds=1)
        print("\n\n" + "="*80)
        print(result)
        print("="*80)
        
        # Test another related query to see if caching helps
        print("\nTesting related query with caching...")
        result2 = system.process_query("What are recent advancements in AI?", max_research_rounds=1)
        print("\n\n" + "="*80)
        print(result2)
        print("="*80)
        
        stats = system.get_research_stats()
        print("\nResearch Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("Research process completed successfully!")
    except Exception as e:
        print(f"Error in research process: {str(e)}")