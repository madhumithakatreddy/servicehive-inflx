AutoStream AI Agent
A conversational AI agent that converts social media conversations into qualified leads using LangGraph and Groq.

## Project Structure

SERVICEHIVE/
├── agent_now/
│   ├── agent.py          # Main LangGraph agent
│   ├── app.py            # CLI chat interface
│   ├── knowledge.json    # AutoStream pricing & policies
│   ├── rag.py            # Knowledge base retrieval
│   └── tools.py          # Mock lead capture tool
├── .env                  # API keys (not committed)
├── .gitignore
├── .python-version
├── pyproject.toml        # Dependencies
├── README.md
└── uv.lock



Setup
 1. Clone the repository
    git clone <your-repo-url>
    cd servicehive

 2. Install dependencies
    uv sync

 3. Add your API key
    Create a .env file in the root directory:
    GROQ_API_KEY=your_groq_api_key_here
    Get your free API key at: https://console.groq.com

 4. Run the agent
    cd agent_now
    uv run python app.py

2. Architecture Explanation (≈200 words): 
   2.1- Langgraph over langchain because-
    1. Multi-Intent Branching
    The agent handles 4 intents — greeting, inquiry, high-intent, and ending — each needing a different path. LangChain supports only linear pipelines with no native conditional branching.
    2. Structured State Management
    LangGraph's typed AgentState persists intent, name, email, and platform reliably across turns. LangChain has no built-in cross-turn state — everything must be managed manually.

   2.2- Architecture Overview-
    1. Graph Structure
    The agent is built as a directed graph with 6 nodes — classify_intent, casual_reply, rag_node, collect_lead, capture_lead, and ending_node.
    2. Shared State
    A typed AgentState dictionary carries messages, intent, name, email, platform, and captured status across every node.
    3. Intent Routing
    Every user message first hits classify_intent, which routes to the correct node using conditional edges.
    4. RAG Pipeline
    rag_node reads knowledge.json, formats it as a string, and injects it into the LLM system prompt as context.
    5. Lead Collection Loop
    collect_lead runs across multiple turns, extracting details one by one until all 3 fields are filled.
    6. Gated Tool Execution
    route_lead_collection checks all 3 fields before allowing capture_lead to fire mock_lead_capture.

   2.3- State Management-
    1. Typed Dictionary
    AgentState holds messages, intent, name, email, platform, and captured — every node reads from and writes to this shared object.
    2. Automatic Merging
    LangGraph merges each node's return value into existing state — nodes only return what changed, not the full state.
    3. Conversation History
    add_messages annotation appends new messages across turns instead of replacing them, preserving full chat history.
    4. Persistent Across Turns
    State is passed into agent.invoke() every turn, carrying all previously collected fields forward.
    5. Gradual Lead Collection
    name, email, and platform start as None and fill one per turn until all three are confirmed

3. WhatsApp Deployment Question:
    1. Register WhatsApp Business API
    Create a Meta Developer account, set up a WhatsApp Business App, and obtain a phone number ID and access token.
    2. Build a Webhook Endpoint
    Create a FastAPI endpoint that receives incoming WhatsApp messages, extracts the user text and sender number, runs it through the agent, and sends the reply back.
    3. Session Management
    Store each user's AgentState in Redis keyed by their WhatsApp number — so conversation memory persists across messages from the same user.
    4. Expose the Webhook
    Use ngrok during development to expose the local server. In production, deploy to a cloud server (AWS/GCP) with HTTPS — required by Meta.
    5. Register Webhook URL
    In Meta Developer Console, register the endpoint URL and verify it with a challenge token. WhatsApp will then forward all incoming messages to it.
    6. Send Reply Back
    After the agent responds, use the WhatsApp Cloud API to send the reply back to the user's number.
 