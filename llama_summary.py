# llama_summary.py
import requests

def summarize_with_llama(notes_text, stats):
    prompt = f"""
You are a helpful assistant that generates **crisp, bullet-point summaries** of a field rep's daily activity for a manager.

Summarize the notes below using bullet points grouped by each lead visited, and then provide a short stats summary at the end.

Make the summary concise, clear, and well-structured.

---
📝 **Notes**:
{notes_text}

📊 **Stats**:
{stats}

✅ Return only the final summary. No explanation needed.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
