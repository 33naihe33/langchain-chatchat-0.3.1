"""Request/response payload logging, independent of optional RAG timing."""
import json

from chatchat.utils import build_logger

logger = build_logger()


def log_payload(endpoint, event, request_id, payload):
    logger.info("CHAT_PAYLOAD {}", json.dumps({
        "endpoint": endpoint, "event": event,
        "request_id": request_id, "payload": payload,
    }, ensure_ascii=False))
