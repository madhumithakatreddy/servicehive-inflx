import json
import re
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rag import get_knowledge_context
from tools import mock_lead_capture

load_dotenv()

class AgentState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    intent:    Optional[str]
    name:      Optional[str]
    email:     Optional[str]
    platform:  Optional[str]
    captured:  bool

tools = [mock_lead_capture]
llm       = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_tools = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)


def classify_intent(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content.strip().lower()

    # Check for ending FIRST before anything else
    ending_words = ["bye", "goodbye", "thanks", "thank you", "that's it", 
                    "see you", "done", "exit", "quit", "stop"]
    if any(word in last_msg for word in ending_words):
        print(f"\n[Intent detected: ending]")
        return {"intent": "ending"}

    # If we're already collecting lead details, don't reclassify
    if state.get("name") or state.get("email") or state.get("platform"):
        return {"intent": "high_intent"}

    # If intent was already high, keep it
    if state.get("intent") == "high_intent":
        return {"intent": "high_intent"}

    last_msg = state["messages"][-1].content
    system = SystemMessage(content="""
You are an intent classifier for AutoStream, a SaaS video editing tool.
Classify the user message into EXACTLY one of these intents:

- "greeting"    : casual hello, hi, hey, how are you
- "inquiry"     : asking about features, pricing, plans, policies, refunds. Also short acknowledgements like "okay", "I see", "got it", "alright"
- "high_intent" : wants to sign up, try, buy, subscribe, start, get the plan
- "ending"      : explicit goodbye only — bye, goodbye, thank you, thanks, that's it, see you, done, exit. NOT "okay" alone.

Reply with ONLY the intent word. Nothing else.
""")
    response = llm.invoke([system, HumanMessage(content=last_msg)])
    intent = response.content.strip().lower().replace('"', '').replace("'", "")

    if intent not in ["greeting", "inquiry", "high_intent", "ending"]:
        intent = "inquiry"

    print(f"\n[Intent detected: {intent}]")
    return {"intent": intent}


def casual_reply(state: AgentState) -> AgentState:
    system = SystemMessage(content="""
You are a conversational AI assistant for AutoStream, an automated video editing SaaS for content creators.
Goal: To understand user needs and convert interested users into qualified leads.

RULES:
1. Respond naturally to what the user said in 2-3 words (e.g. "I'm doing great!", "Nice to meet you!")
2. Then in a new sentence ask how you can help them today.
3. Do not say anything else.
4. Do NOT make up any features, offers, or pricing.
""")
    response = llm.invoke([system] + list(state["messages"]))
    return {"messages": [AIMessage(content=response.content)]}


def rag_node(state: AgentState) -> AgentState:
    context = get_knowledge_context()
    system = SystemMessage(content=f"""
You are a helpful conversational AI assistant

STRICT RULES:
1. Use ONLY the information provided in below.
2. Do NOT add any offers, discounts, promotions, or features not mentioned below.
3. Do NOT say "let me check" or "I'll look into that" — you only know what's below.
4. Do NOT make anything up or speculate.
5. If the user asks something not covered below, say exactly:
   "I don't have that information, please contact our support team."
6. Do Not mention promotions, discounts, or deals unless explicitly in the knowledge base.

KNOWLEDGE BASE:
{context}
""")
    response = llm.invoke([system] + list(state["messages"]))
    return {"messages": [AIMessage(content=response.content)]}


def collect_lead(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content
    updates = extract_lead_details(state, last_msg)
    name     = updates.get("name")     or state.get("name")
    email    = updates.get("email")    or state.get("email")
    platform = updates.get("platform") or state.get("platform")
    system = SystemMessage(content=f"""
You are collecting sign-up details for AutoStream.

Current details collected:
- Name:     {name or 'NOT collected yet'}
- Email:    {email or 'NOT collected yet'}
- Platform: {platform or 'NOT collected yet'}

STRICT RULES:
1. Ask for the NEXT missing detail only — one at a time in this order:
   1. Name
   2. Email address
   3. Creator platform (YouTube, Instagram, TikTok, etc.)
2. Do NOT mention Google account, Facebook, or any login options.
3. Do NOT ask for password, phone number, or anything else.
4. Do NOT repeat back what the user already said.
5. Just ask only for the missing field simply and briefly
6. NEVER make up any steps or account creation process.

Be friendly and conversational. Do NOT ask for details already collected.
If all 3 are collected, say "Great! Let me get you signed up." and nothing else.
""")
    response = llm.invoke([system] + list(state["messages"]))
    updates["messages"] = [AIMessage(content=response.content)]
    return updates


def extract_lead_details(state: AgentState, user_msg: str) -> dict:
    system = SystemMessage(content="""
Extract lead details from the user message.
Return a JSON object with keys: name, email, platform.
Use null for anything not mentioned.
Return ONLY valid JSON, nothing else.

Examples:
- "I'm John" → {"name": "John", "email": null, "platform": null}
- "my email is a@b.com" → {"name": null, "email": "a@b.com", "platform": null}
- "I use YouTube" → {"name": null, "email": null, "platform": "YouTube"}
""")
    response = llm.invoke([system, HumanMessage(content=user_msg)])
    updates = {}
    try:
        raw = re.sub(r"```json|```", "", response.content).strip()
        extracted = json.loads(raw)
        if extracted.get("name")     and not state.get("name"):
            updates["name"]     = extracted["name"]
        if extracted.get("email")    and not state.get("email"):
            updates["email"]    = extracted["email"]
        if extracted.get("platform") and not state.get("platform"):
            updates["platform"] = extracted["platform"]
    except Exception:
        pass
    return updates


def capture_lead_node(state: AgentState) -> AgentState:
    result = mock_lead_capture.invoke({
        "name":     state['name'],
        "email":    state['email'],
        "platform": state['platform']
    })
    
    return {
        "messages": [AIMessage(content=result)],
        "captured": True
    }


def ending_node(state: AgentState) -> AgentState:
    system = SystemMessage(content="""
You are a conversational AI assistant for AutoStream.
The user is ending the conversation.

RULES:
1. Thank the user for their time in ONE sentence.
2. Let them know AutoStream is always here if they need help.
3. Say goodbye warmly.
4. Keep it to 2 sentences max.
5. Do not say anything else.
""")
    response = llm.invoke([system] + list(state["messages"]))
    return {"messages": [AIMessage(content=response.content)]}


def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "inquiry")
    if intent == "greeting":
        return "casual_reply"
    elif intent == "high_intent":
        return "collect_lead"
    elif intent == "ending":
        return "ending_node"
    else:
        return "rag_node"


def route_lead_collection(state: AgentState) -> str:
    if state.get("name") and state.get("email") and state.get("platform"):
        return "capture"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("casual_reply",    casual_reply)
    graph.add_node("rag_node",        rag_node)
    graph.add_node("collect_lead",    collect_lead)
    graph.add_node("capture_lead",    capture_lead_node)
    graph.add_node("ending_node",     ending_node)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "casual_reply": "casual_reply",
            "rag_node":     "rag_node",
            "collect_lead": "collect_lead",
            "ending_node":  "ending_node",
        }
    )
    graph.add_edge("casual_reply", END)
    graph.add_edge("rag_node",     END)
    graph.add_edge("ending_node",  END)
    graph.add_conditional_edges(
        "collect_lead",
        route_lead_collection,
        {
            "capture": "capture_lead",
            "continue": END,
        }
    )
    graph.add_edge("capture_lead", END)
    return graph.compile()


agent = build_graph()
