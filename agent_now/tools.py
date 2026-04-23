from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Captures a qualified lead.
    Only call this tool when you have ALL THREE details:
    name, email, and platform.
    """
    print(f"\n{'='*50}")
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead successfully captured for {name} ({email}) on {platform}."