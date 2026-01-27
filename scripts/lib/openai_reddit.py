"""OpenAI Responses API client for Reddit discovery."""

import json
import re
import sys
from typing import Any, Dict, List, Optional

from . import http, env


def _log_error(msg: str):
    """Log error to stderr."""
    sys.stderr.write(f"[REDDIT ERROR] {msg}\n")
    sys.stderr.flush()

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_INSTRUCTIONS = (
    "You are a research assistant for a skill that summarizes what people are "
    "discussing in the last 30 days. Your goal is to find relevant Reddit threads "
    "about the topic and return ONLY the required JSON. Be inclusive (return more "
    "rather than fewer), but avoid irrelevant results. Prefer threads with discussion "
    "and comments. If you can infer a date, include it; otherwise use null. "
    "Do not include developers.reddit.com or business.reddit.com."
)


def _parse_sse_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE chunk into a JSON object."""
    lines = chunk.split("\n")
    data_lines = []

    for line in lines:
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return None

    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _parse_sse_stream_raw(raw: str) -> List[Dict[str, Any]]:
    """Parse SSE stream from raw text and return JSON events."""
    events: List[Dict[str, Any]] = []
    buffer = ""
    for chunk in raw.splitlines(keepends=True):
        buffer += chunk
        while "\n\n" in buffer:
            event_chunk, buffer = buffer.split("\n\n", 1)
            event = _parse_sse_chunk(event_chunk)
            if event is not None:
                events.append(event)
    if buffer.strip():
        event = _parse_sse_chunk(buffer)
        if event is not None:
            events.append(event)
    return events


def _parse_codex_stream(raw: str) -> Dict[str, Any]:
    """Parse SSE stream from Codex responses into a response-like dict."""
    events = _parse_sse_stream_raw(raw)

    # Prefer explicit completed response payload if present
    for evt in reversed(events):
        if isinstance(evt, dict):
            if evt.get("type") == "response.completed" and isinstance(evt.get("response"), dict):
                return evt["response"]
            if isinstance(evt.get("response"), dict):
                return evt["response"]

    # Fallback: reconstruct output text from deltas
    output_text = ""
    for evt in events:
        if not isinstance(evt, dict):
            continue
        delta = evt.get("delta")
        if isinstance(delta, str):
            output_text += delta
            continue
        text = evt.get("text")
        if isinstance(text, str):
            output_text += text

    if output_text:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ]
        }

    return {}

# Depth configurations: (min, max) threads to request
# Request MORE than needed since many get filtered by date
DEPTH_CONFIG = {
    "quick": (15, 25),
    "default": (30, 50),
    "deep": (70, 100),
}

REDDIT_SEARCH_PROMPT = """Find Reddit discussion threads about: {topic}

STEP 1: EXTRACT THE CORE SUBJECT
Get the MAIN NOUN/PRODUCT/TOPIC:
- "best nano banana prompting practices" → "nano banana"
- "killer features of clawdbot" → "clawdbot"
- "top Claude Code skills" → "Claude Code"
DO NOT include "best", "top", "tips", "practices", "features" in your search.

STEP 2: SEARCH BROADLY
Search for the core subject:
1. "[core subject] site:reddit.com"
2. "reddit [core subject]"
3. "[core subject] reddit"

Return as many relevant threads as you find. We filter by date server-side.

STEP 3: INCLUDE ALL MATCHES
- Include ALL threads about the core subject
- Set date to "YYYY-MM-DD" if you can determine it, otherwise null
- We verify dates and filter old content server-side
- DO NOT pre-filter aggressively - include anything relevant

REQUIRED: URLs must contain "/r/" AND "/comments/"
REJECT: developers.reddit.com, business.reddit.com

Find {min_items}-{max_items} threads. Return MORE rather than fewer.

Return JSON:
{{
  "items": [
    {{
      "title": "Thread title",
      "url": "https://www.reddit.com/r/sub/comments/xyz/title/",
      "subreddit": "subreddit_name",
      "date": "YYYY-MM-DD or null",
      "why_relevant": "Why relevant",
      "relevance": 0.85
    }}
  ]
}}"""


def _extract_core_subject(topic: str) -> str:
    """Extract core subject from verbose query for retry."""
    noise = ['best', 'top', 'how to', 'tips for', 'practices', 'features',
             'killer', 'guide', 'tutorial', 'recommendations', 'advice',
             'prompting', 'using', 'for', 'with', 'the', 'of', 'in', 'on']
    words = topic.lower().split()
    result = [w for w in words if w not in noise]
    return ' '.join(result[:3]) or topic  # Keep max 3 words


def _build_payload(model: str, instructions_text: str, input_text: str, auth_source: str) -> Dict[str, Any]:
    """Build responses payload for OpenAI or Codex endpoints."""
    payload = {
        "model": model,
        "store": False,
        "tools": [
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": ["reddit.com"]
                }
            }
        ],
        "include": ["web_search_call.action.sources"],
        "instructions": instructions_text,
        "input": input_text,
    }
    if auth_source == env.AUTH_SOURCE_CODEX:
        payload["input"] = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_text}],
            }
        ]
        payload["stream"] = True
    return payload


def search_reddit(
    api_key: str,
    model: str,
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    auth_source: str = "api_key",
    account_id: Optional[str] = None,
    mock_response: Optional[Dict] = None,
    _retry: bool = False,
) -> Dict[str, Any]:
    """Search Reddit for relevant threads using OpenAI Responses API.

    Args:
        api_key: OpenAI API key
        model: Model to use
        topic: Search topic
        from_date: Start date (YYYY-MM-DD) - only include threads after this
        to_date: End date (YYYY-MM-DD) - only include threads before this
        depth: Research depth - "quick", "default", or "deep"
        mock_response: Mock response for testing

    Returns:
        Raw API response
    """
    if mock_response is not None:
        return mock_response

    min_items, max_items = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])

    if auth_source == env.AUTH_SOURCE_CODEX:
        if not account_id:
            raise ValueError("Missing chatgpt_account_id for Codex auth")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
            "Content-Type": "application/json",
        }
        url = CODEX_RESPONSES_URL
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = OPENAI_RESPONSES_URL

    # Adjust timeout based on depth (generous for OpenAI web_search which can be slow)
    timeout = 90 if depth == "quick" else 120 if depth == "default" else 180

    # Note: allowed_domains accepts base domain, not subdomains
    # We rely on prompt to filter out developers.reddit.com, etc.
    instructions_text = (
        CODEX_INSTRUCTIONS
        + "\n\n"
        + REDDIT_SEARCH_PROMPT.format(
            topic=topic,
            from_date=from_date,
            to_date=to_date,
            min_items=min_items,
            max_items=max_items,
        )
    )
    input_text = topic
    payload = _build_payload(model, instructions_text, input_text, auth_source)

    if auth_source == env.AUTH_SOURCE_CODEX:
        raw = http.post_raw(url, payload, headers=headers, timeout=timeout)
        response = _parse_codex_stream(raw or "")
    else:
        response = http.post(url, payload, headers=headers, timeout=timeout)
    return response


def parse_reddit_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse OpenAI response to extract Reddit items.

    Args:
        response: Raw API response

    Returns:
        List of item dicts
    """
    items = []

    # Check for API errors first
    if "error" in response and response["error"]:
        error = response["error"]
        err_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        _log_error(f"OpenAI API error: {err_msg}")
        if http.DEBUG:
            _log_error(f"Full error response: {json.dumps(response, indent=2)[:1000]}")
        return items

    # Try to find the output text
    output_text = ""
    if "output" in response:
        output = response["output"]
        if isinstance(output, str):
            output_text = output
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                output_text = c.get("text", "")
                                break
                    elif "text" in item:
                        output_text = item["text"]
                elif isinstance(item, str):
                    output_text = item
                if output_text:
                    break

    # Also check for choices (older format)
    if not output_text and "choices" in response:
        for choice in response["choices"]:
            if "message" in choice:
                output_text = choice["message"].get("content", "")
                break

    if not output_text:
        print(f"[REDDIT WARNING] No output text found in OpenAI response. Keys present: {list(response.keys())}", flush=True)
        return items

    # Extract JSON from the response
    json_match = re.search(r'\{[\s\S]*"items"[\s\S]*\}', output_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            items = data.get("items", [])
        except json.JSONDecodeError:
            pass

    # Validate and clean items
    clean_items = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        url = item.get("url", "")
        if not url or "reddit.com" not in url:
            continue

        clean_item = {
            "id": f"R{i+1}",
            "title": str(item.get("title", "")).strip(),
            "url": url,
            "subreddit": str(item.get("subreddit", "")).strip().lstrip("r/"),
            "date": item.get("date"),
            "why_relevant": str(item.get("why_relevant", "")).strip(),
            "relevance": min(1.0, max(0.0, float(item.get("relevance", 0.5)))),
        }

        # Validate date format
        if clean_item["date"]:
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(clean_item["date"])):
                clean_item["date"] = None

        clean_items.append(clean_item)

    return clean_items
