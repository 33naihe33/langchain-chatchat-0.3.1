# SSE TTFT / TPOT Benchmark Script

## Purpose

Provide a small command-line benchmark for the existing `POST /chat/kb_chat` SSE endpoint. It measures MiniCPM5-2B's user-visible response latency after a short warm-up.

## Scope

- Send two warm-up requests that are excluded from metrics.
- Send five measured streaming requests sequentially.
- Treat the first SSE event with non-empty `choices[0].delta.content` as the first generated text.
- Report per-run TTFT, generation duration, output text and output-token count.
- When a compatible local tokenizer directory is supplied, calculate TPOT as `(stream_end - first_content) / max(output_tokens - 1, 1)`.
- Report mean, P50 and P95 for TTFT and TPOT across measured runs.

## Interface

The script accepts command-line options for the service URL, query, knowledge-base name, model name, request count, warm-up count and optional tokenizer path. Defaults target the project’s documented `/chat/kb_chat` endpoint and use two warm-ups plus five measurements.

## Error Handling

- Reject non-2xx HTTP responses and malformed SSE JSON with a clear error.
- Reject a completed response with no non-empty generated content.
- Continue without TPOT when no tokenizer path is provided, while still emitting TTFT and generation duration.

## Verification

Unit tests will cover SSE event parsing, first-content timing selection, TPOT calculation and percentile summarization. The script itself can be run against a live service after the server and knowledge base are available.
