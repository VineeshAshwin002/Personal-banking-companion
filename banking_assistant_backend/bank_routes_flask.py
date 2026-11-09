"""
Flask SSE routes for the banking LangGraph application.

Install:
    pip install flask flask-cors

Run:
    python bank_routes_flask.py

API:
    POST http://127.0.0.1:8000/chat
    JSON: {"message": "Show my account balance", "customer_id": "C101"}
    The customer_id field is optional for general questions.

Note:
    The graph.invoke() call finishes before answer chunks are emitted. The SSE
    response is chunked for the frontend, but this is not live LLM-token streaming.
"""

import json
import logging
from typing import Any, Dict

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from langchain_core.messages import AIMessage

from bank_multiagent_frontend_ready import graph

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
    ]}},
    supports_credentials=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_final_answer(result: Dict[str, Any]) -> str:
    """Return the last non-empty AI message that is not requesting a tool."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                answer = "".join(text_parts).strip()
                if answer:
                    return answer
    return "I couldn't generate a response for that question."


def sse(event: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = body.get("message")
    customer_id = body.get("customer_id")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "The 'message' field is required and must be a non-empty string."}), 400

    if customer_id is None:
        customer_id = ""
    elif not isinstance(customer_id, str):
        return jsonify({
            "error": "The 'customer_id' field must be a string when provided."
        }), 400

    user_message = message.strip()
    customer_id = customer_id.strip()

    @stream_with_context
    def generate():
        yield sse("start", {"message": "Request received"})
        try:
            initial_state = {
                "agents": [],
                "user_request": user_message,
                "messages": [],
                "security": "",
                "user_details": {},
                "customer_id": customer_id,
            }

            # Graph execution completes before response chunks are emitted.
            result = graph.invoke(initial_state)
            answer = get_final_answer(result)

            yield sse("question", {"message": user_message})
            chunk_size = 40
            for index in range(0, len(answer), chunk_size):
                yield sse("chunk", {"text": answer[index:index + chunk_size]})

            yield sse("complete", {"answer": answer})
        except Exception:
            logger.exception("Failed to process /chat request")
            yield sse("error", {
                "message": "Unable to process the request. Check the Flask API logs."
            })

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
