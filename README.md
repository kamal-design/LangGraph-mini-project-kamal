# LangGraph-mini-project-kamal

## 📚 Learning RAG, LangGraph with Tools Calls, CoT, ReAct, Customization, Memory & Tavily Web Search

This repository contains a comprehensive learning project exploring **LangGraph**, **RAG (Retrieval-Augmented Generation)**, **Chain-of-Thought (CoT)**, **ReAct patterns**, **customization**, **memory management**, and **Tavily web search integration** using Python and various AI frameworks.

---

## 🎯 Project Overview

This project is a hands-on learning journey through modern AI agent development patterns, focusing on:

- **LangGraph**: A framework for building stateful, multi-actor applications with LLMs
- **ReAct**: Reasoning and Acting pattern for tool-using AI agents
- **RAG**: Retrieval-Augmented Generation for knowledge-enhanced AI
- **Chain-of-Thought (CoT)**: Structured reasoning patterns
- **Memory Management**: Conversation persistence with in-memory and external databases
- **Tool Calls**: Integrating external functions and APIs with LLMs
- **Customization**: Building custom nodes and workflows
- **Tavily Web Search**: Real-time information retrieval from the web

---

## 📁 Project Structure

```
LangGraph-mini-project/
├── README.md                          # Project documentation
├── steps.txt                          # Setup and development instructions
├── .venv/                             # Python virtual environment
├── state_db/                          # SQLite database for external memory
│   └── example.db                     # Pre-populated database file
└── colab-notebooks/
    ├── LangGraph_RaAct.ipynb           # ReAct pattern with tool calls
    ├── LangGraph_Router.ipynb          # Conditional routing workflows
    ├── LangGraph_memory.ipynb          # In-memory conversation persistence
    ├── LangGraph_memory_externalDB.ipynb # SQLite database memory
    └── LangGraph_tavily_websearch.ipynb # Web search with Tavily & RAG
```

---

## 🧪 Colab Notebooks Description

### 1. **LangGraph_RaAct.ipynb** - Reasoning and Acting with Tool Calls

**Focus**: ReAct pattern implementation using LangGraph

**Key Concepts**:
- **ReAct Framework**: Combines reasoning traces and tool actions
- **Tool Integration**: Creating and binding custom tools (add, multiply, divide)
- **LangGraph Workflow**: Building graphs with nodes and edges
- **State Management**: Using MessagesState for conversation tracking

**Implementation Details**:
- Uses `langchain-groq` with OpenAI GPT-OSS-120B model
- Implements arithmetic tools with proper docstrings for ReAct compatibility
- Creates a StateGraph with assistant and tool nodes
- Uses `tools_condition` for conditional edge routing
- Visualizes the graph using mermaid diagrams

**Example Use Case**: 
```python
# Tool calls for arithmetic operations
messages = [HumanMessage(content="Add 124133 and 213214. Multiply the output by 2. Divide the output by 5")]
result = react_graph.invoke({"messages": messages})
# Result: 134,938.8
```

---

### 2. **LangGraph_Router.ipynb** - Conditional Routing Workflows

**Focus**: Building decision-making workflows with conditional edges

**Key Concepts**:
- **State Management**: Custom TypedDict for journey state
- **Conditional Edges**: Dynamic routing based on conditions
- **Random Decision Making**: Probabilistic path selection
- **Message Types**: Working with HumanMessage and AIMessage

**Implementation Details**:
- Creates a journey planner from Coimbatore to Trichy/Salem
- Uses `decide_intermediate_city` function with random selection
- Demonstrates START and END states in StateGraph
- Shows how to track conversation history with message objects

**Example Use Case**:
```python
# Journey with conditional routing
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"current_location": "Coimbatore"}, config)
# Randomly routes to either Trichy or Salem
```

---

### 3. **LangGraph_memory.ipynb** - Conversation Memory with MemorySaver

**Focus**: In-memory conversation persistence

**Key Concepts**:
- **MemorySaver**: LangGraph's built-in memory checkpointing
- **Thread-based Conversations**: Maintaining context across turns
- **State Persistence**: Preserving conversation history
- **Configuration**: Using thread_id for conversation tracking

**Implementation Details**:
- Compares behavior with and without memory
- Shows how memory preserves context between messages
- Demonstrates `MemorySaver` checkpointing
- Visualizes the difference in responses with/without memory

**Example Use Case**:
```python
# Without memory - loses context
messages = [HumanMessage(content="Add 44 and 22")]
result = react_graph.invoke({"messages": messages})
# Result: 66

messages = [HumanMessage(content="Multiply that by 3")]
result = react_graph.invoke({"messages": messages})
# Without memory: "I'm not sure which value you'd like to multiply by 3"

# With memory - maintains context
react_graph_memory = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
messages = [HumanMessage(content="Add 44 and 22")]
result = react_graph_memory.invoke({"messages": messages}, config)
# Result: 66

messages = [HumanMessage(content="Multiply that by 3")]
result = react_graph_memory.invoke({"messages": messages}, config)
# With memory: "Multiplying 66 by 3 gives 198"
```

---

### 4. **LangGraph_memory_externalDB.ipynb** - External SQLite Database Memory

**Focus**: Persistent memory using external databases

**Key Concepts**:
- **SqliteSaver**: Database-backed checkpointing
- **External State Management**: Using SQLite for conversation persistence
- **Database Schema**: Understanding checkpoints and writes tables
- **Thread Isolation**: Multiple conversations in one database

**Implementation Details**:
- Installs `langgraph-checkpoint-sqlite` package
- Creates `state_db/example.db` SQLite database
- Uses `SqliteSaver` for persistent memory storage
- Shows database schema with checkpoints and writes tables
- Demonstrates multi-thread conversation management

**Example Use Case**:
```python
# Connect to SQLite database
conn = sqlite3.connect("state_db/example.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# Conversation persists across sessions
input_message = HumanMessage(content="hi! I'm KamalM")
output = graph.invoke({"messages": [input_message]}, config)
# Output: "Nice to meet you, KamalM!"

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config)
# Output: "Your name is KamalM."
```

**Database Tables**:
- `checkpoints`: Stores conversation state snapshots
- `writes`: Tracks individual message writes

---

### 5. **LangGraph_tavily_websearch.ipynb** - Web Search with RAG

**Focus**: Real-time information retrieval and RAG implementation

**Key Concepts**:
- **Tavily Search**: Web search API integration
- **Web Scraping**: Using WebBaseLoader to fetch page content
- **RAG Pipeline**: Retrieval + Generation workflow
- **Document Processing**: Handling scraped web content
- **Search Configuration**: Customizing search parameters

**Implementation Details**:
- Installs `langchain-tavily` and `langchain-community` packages
- Configures TavilySearch with:
  - `max_results`: Number of search results
  - `search_depth`: "basic" or "advanced"
  - `include_domains`: Restrict to specific websites
  - `include_raw_content`: Get full page content
- Implements web_search_node for LangGraph integration
- Scrapes and processes WHO malaria reports as example

**Example Use Case**:
```python
# Configure Tavily search
tavily_search = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_domains=["who.int"],
    include_raw_content=True
)

# Search for information
question = "What is the latest WHO report on malaria?"
search_results = tavily_search.invoke(question)

# Scrape and process results
urls = [result.get("url") for result in search_results["results"]]
for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Process documents for RAG
```

**Search Results**:
- World Malaria Report 2024 (263M cases, 597K deaths in 2023)
- WHO Malaria Fact Sheet (282M cases, 610K deaths in 2024)
- Historical reports and trends

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.12+
- Jupyter Notebook or Google Colab
- Required API keys:
  - **GROQ_API_KEY**: For Groq LLM access (used in all notebooks)
  - **TAVILY_API_KEY**: For Tavily web search (used in websearch notebook)

**Note**: See `steps.txt` for detailed local development setup instructions including virtual environment creation and LangGraph Studio configuration.

### Installation

```bash
# Core packages (used across all notebooks)
pip install langchain-groq langgraph

# Memory packages
pip install langgraph-checkpoint-sqlite aiosqlite

# Web search packages
pip install langchain-tavily langchain-community aiohttp

# For SQLite external DB
pip install sqlite-vec
```

### API Key Setup

```python
import os
import getpass

# Groq API key (used in all notebooks)
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Tavily API key (used in websearch notebook)
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
```

---

## 🎓 Key Learning Outcomes

### 1. **ReAct Pattern** (LangGraph_RaAct.ipynb)
- ✅ Understanding ReAct: Reasoning + Action framework
- ✅ Creating tool-compatible functions with docstrings
- ✅ Binding tools to LLMs using `llm.bind_tools()`
- ✅ Building execution graphs with conditional edges
- ✅ Visualizing workflows with mermaid diagrams

**Key Code Pattern**:
```python
# Define tools with docstrings
def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

# Bind tools to LLM
llm_with_tools = llm.bind_tools([add, multiply, divide])

# Create graph with conditional routing
builder.add_conditional_edges("assistant", tools_condition)
```

---

### 2. **Conditional Routing** (LangGraph_Router.ipynb)
- ✅ Creating custom state types with TypedDict
- ✅ Implementing decision nodes for routing
- ✅ Using START and END states
- ✅ Working with different message types
- ✅ Understanding graph execution flow

**Key Code Pattern**:
```python
class JourneyState(TypedDict):
    current_location: str

def decide_intermediate_city(state: JourneyState) -> Literal["trichy", "salem"]:
    if random.random() < 0.5:
        return "trichy"
    else:
        return "salem"

builder.add_conditional_edges("coimbatore", decide_intermediate_city)
```

---

### 3. **Memory Management** (LangGraph_memory.ipynb)
- ✅ Understanding stateless vs stateful workflows
- ✅ Implementing MemorySaver for in-memory persistence
- ✅ Using thread_id for conversation tracking
- ✅ Preserving context across multiple turns
- ✅ Visualizing memory impact on responses

**Key Code Pattern**:
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": messages}, config)
```

---

### 4. **External Database Memory** (LangGraph_memory_externalDB.ipynb)
- ✅ Setting up SQLite database for persistence
- ✅ Using SqliteSaver for database-backed memory
- ✅ Understanding database schema (checkpoints, writes tables)
- ✅ Managing multiple conversations in one database
- ✅ Querying and inspecting stored states

**Key Code Pattern**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("state_db/example.db", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# View conversation state
graph_state = graph.get_state(config)
```

**Database Structure**:
```sql
-- Checkpoints table stores state snapshots
CREATE TABLE checkpoints (...);

-- Writes table tracks individual operations
CREATE TABLE writes (...);
```

---

### 5. **RAG with Web Search** (LangGraph_tavily_websearch.ipynb)
- ✅ Configuring Tavily search with various parameters
- ✅ Implementing web scraping with WebBaseLoader
- ✅ Building RAG workflows with LangGraph
- ✅ Processing and formatting search results
- ✅ Creating document-based response generation

**Key Code Pattern**:
```python
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader

tavily_search = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_domains=["who.int"],
    include_raw_content=True
)

# Build RAG graph
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    sender: str

def web_search_node(state: GraphState) -> GraphState:
    search_results = tavily_search.invoke(state["question"])
    # Scrape and process results
    return {"documents": scraped_docs, "sender": "web_search_node"}
```

---

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Provider** | Groq API | Fast inference with open-source models |
| **LLM Model** | openai/gpt-oss-120b | Default model used |
| **Framework** | LangGraph | Agent workflow orchestration |
| **LangChain** | langchain-core, langchain-groq | LLM integration |
| **Memory** | MemorySaver, SqliteSaver | Conversation persistence |
| **Search** | Tavily API | Web search capabilities |
| **Scraping** | WebBaseLoader | HTML content extraction |
| **Database** | SQLite | Persistent memory storage |
| **Visualization** | Mermaid | Graph workflow visualization |

---

## 📊 Comparison of Approaches

| Feature | MemorySaver | SqliteSaver | No Memory |
|---------|-------------|-------------|------------|
| **Persistence** | In-memory | Database | None |
| **Survives Restart** | ❌ No | ✅ Yes | ❌ No |
| **Multi-session** | ❌ No | ✅ Yes | ❌ No |
| **Setup Complexity** | Low | Medium | None |
| **Performance** | Fast | Medium | Fast |
| **Use Case** | Single session | Production | Stateless |

| Feature | ReAct | Router | RAG |
|---------|--------|---------|-----|
| **Tool Use** | ✅ Yes | ❌ No | ✅ Yes |
| **Decision Making** | Tool-based | Conditional | Search-based |
| **State Complexity** | Medium | Low | High |
| **External Data** | ❌ No | ❌ No | ✅ Yes |
| **Use Case** | Calculations, APIs | Workflows | Knowledge Q&A |

---

## 🎯 Use Cases Covered

### ReAct Pattern
- **Mathematical Calculations**: Chained arithmetic operations
- **API Integration**: Calling external services
- **Data Processing**: Multi-step data transformations

### Router Pattern
- **Decision Trees**: Multi-path workflows
- **Journey Planning**: Location-based routing
- **Conditional Logic**: Probabilistic decision making

### Memory Patterns
- **Chatbots**: Conversation history preservation
- **Personal Assistants**: User preference tracking
- **Multi-turn Dialogues**: Context-aware conversations

### RAG Pattern
- **Knowledge Q&A**: Answering questions from web sources
- **Current Events**: Real-time information retrieval
- **Research Assistant**: Gathering and synthesizing information

---

## 🏆 Best Practices Learned

### 1. **Tool Design**
- Always include comprehensive docstrings for ReAct compatibility
- Use type hints for better error handling
- Keep tools focused on single responsibilities
- Validate inputs and handle edge cases

### 2. **Graph Design**
- Use meaningful node names
- Document edge conditions clearly
- Visualize complex workflows with mermaid
- Keep graphs focused on specific use cases

### 3. **Memory Management**
- Use thread_id to isolate conversations
- Choose MemorySaver for development, SqliteSaver for production
- Consider cleanup strategies for old conversations
- Monitor memory usage for performance

### 4. **Web Search**
- Restrict domains for focused searches
- Use advanced search depth for comprehensive results
- Implement error handling for scraping failures
- Cache results to avoid redundant searches

### 5. **Error Handling**
- Handle API rate limits gracefully
- Validate search results before processing
- Implement retry logic for transient failures
- Provide meaningful error messages to users

---

## 🔮 Future Enhancements

### Planned Additions:
- [ ] **Multi-modal RAG**: Adding image and video search capabilities
- [ ] **Vector Database**: Integrating FAISS or Chroma for document similarity
- [ ] **Custom Tools**: Building specialized tool sets for different domains
- [ ] **Evaluation**: Adding metrics for measuring agent performance
- [ ] **Deployment**: Docker and FastAPI integration for production
- [ ] **Monitoring**: Logging and observability for production systems
- [ ] **Security**: Authentication and authorization for API endpoints

### Advanced Patterns to Explore:
- **Hierarchical Agents**: Agents that call other agents
- **Parallel Execution**: Running multiple tools simultaneously
- **Fallback Mechanisms**: Graceful degradation when tools fail
- **Cost Optimization**: Minimizing API calls and token usage
- **Custom Checkpointing**: Custom memory serialization formats

---

## 📚 Additional Resources

### Official Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Groq API Documentation](https://console.groq.com/docs)
- [Tavily API Documentation](https://docs.tavily.com/)

### Learning Materials
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [RAG Survey](https://arxiv.org/abs/2305.18567)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Groq Discord](https://discord.gg/groq)

---

## 💡 Key Takeaways

1. **LangGraph provides powerful abstractions** for building complex AI workflows with clean, modular code

2. **ReAct pattern** enables LLMs to reason about when and how to use tools, creating more reliable and transparent agents

3. **Memory is crucial** for multi-turn conversations - choose the right persistence strategy for your use case

4. **RAG with web search** brings real-time knowledge to your AI applications, but requires careful error handling

5. **Conditional routing** allows you to build decision-making workflows that adapt based on inputs and state

6. **Tool integration** is straightforward with proper docstrings and type hints, enabling natural language tool invocation

7. **Visualization** is key to understanding and debugging complex workflows - use mermaid diagrams liberally

---

## 📝 Project Timeline

Based on git commits:
- **2024-10-07**: LangGraph Router notebook created
- **2024-10-09**: LangGraph ReAct notebook added
- **2024-10-14**: Memory notebook with local storage
- **2024-10-23**: External SQLite memory notebook added
- **2024-10-23**: Tavily web search notebook completed

---

## 🎉 Conclusion

This project provides a **comprehensive, hands-on guide** to building AI agents with LangGraph, covering all the essential patterns and techniques:

- ✅ **ReAct** for tool-using agents
- ✅ **Conditional routing** for decision-making workflows
- ✅ **Memory management** for conversation persistence
- ✅ **RAG** for knowledge-enhanced AI
- ✅ **Web search** for real-time information

Each notebook builds on the previous ones, creating a complete learning path from basic tool calls to sophisticated RAG systems with persistent memory.

**Perfect for**: Developers, researchers, and AI enthusiasts looking to understand how to build production-ready AI agents with modern frameworks.

---

*Generated from Colab Notebooks in: /Volumes/work/kamal/LangGraph-mini-project/colab-notebooks/*

*Last Updated: October 23, 2026*
