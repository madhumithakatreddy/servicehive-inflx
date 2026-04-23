import json
import os

# Load knowledge base from JSON file
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")

def load_knowledge_base() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)

def get_knowledge_context() -> str:
    """
    Reads the knowledge base and returns it as a
    formatted string to inject into the LLM prompt.
    """
    kb = load_knowledge_base()

    context = f"""
Company: {kb['company']}
About: {kb['description']}

=== PLANS & PRICING ===

Basic Plan:
- Price: {kb['plans']['basic']['price']}
- Features: {', '.join(kb['plans']['basic']['features'])}

Pro Plan:
- Price: {kb['plans']['pro']['price']}
- Features: {', '.join(kb['plans']['pro']['features'])}

=== POLICIES ===
- Refund: {kb['policies']['refund']}
- Support: {kb['policies']['support']}
"""
    return context.strip()
