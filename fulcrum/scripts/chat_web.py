#!/usr/bin/env python3
"""
fulcrum chat: Interactive Web Chat & Real-Time Telemetry Server
Spins up a local web server to chat with open-weight models deployed on GCP
(Vertex AI Endpoints or Compute Engine vLLM instances) while managing Google Cloud
IAM tokens and displaying live inference statistics (TTFT, tokens/sec throughput).
"""

import argparse
import html
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

# Import local state database utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import db  # noqa: E402


class TokenManager:
    """
    Manages Google Cloud IAM OAuth2 access tokens.
    Caches tokens with thread-safe locking and handles automatic refresh.
    """

    def __init__(self, ttl_seconds: int = 3000):
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._cached_at: float = 0.0
        self._ttl = ttl_seconds

    def get_token(self, force_refresh: bool = False) -> Optional[str]:
        """Retrieves a cached token or requests a fresh token from gcloud."""
        env_token = os.environ.get("GCP_TOKEN") or os.environ.get("BEARER_TOKEN")
        if env_token:
            return env_token

        with self._lock:
            now = time.time()
            if not force_refresh and self._token and (now - self._cached_at < self._ttl):
                return self._token

            try:
                # Execute gcloud auth print-access-token safely
                output = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).strip()
                if output:
                    self._token = output
                    self._cached_at = now
                    return self._token
            except Exception:
                pass
            return self._token


TOKEN_MANAGER = TokenManager()


def resolve_deployment_target(
    identifier: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Resolves target URL, model name, and metadata for chat inference.
    Supports deployment IDs from deployments.json as well as direct endpoints.
    """
    dep = None
    if identifier:
        dep = db.get_deployment(identifier)

    if not dep and not identifier and not endpoint:
        # Default to the most recently updated deployment if any exist
        all_deps = db.load_deployments()
        if all_deps:
            dep = all_deps[0]

    if dep:
        model_name = dep.get("model") or model or "Unknown Model"
        platform = dep.get("platform", "vertex")
        if platform == "vertex":
            proj = dep.get("project") or db.detect_gcp_project() or "unknown"
            reg = dep.get("region", "us-central1")
            endpoint_name = dep.get("endpoint_numeric_id") or dep.get("endpoint_name") or dep.get("id")
            endpoint_name = dep.get("endpoint_id") or dep.get("endpoint_numeric_id") or dep.get("endpoint_name") or dep.get("id")
            target_url = f"https://{reg}-aiplatform.googleapis.com/v1/projects/{proj}/locations/{reg}/endpoints/{endpoint_name}:rawPredict"
        else:
            ip = dep.get("ip") or "127.0.0.1"
            ip = dep.get("ip_address") or dep.get("ip") or "127.0.0.1"
            port = dep.get("port", 8000)
            target_url = f"http://{ip}:{port}/v1/chat/completions"
        return target_url, model_name, dep.get("id"), dep

    if not endpoint:
        # Fallback to local default if no deployment is found
        return "http://127.0.0.1:8000/v1/chat/completions", model or "local-model", None, None

    model_name = model or "custom-model"
    if "aiplatform.googleapis.com" in endpoint:
        target_url = endpoint if endpoint.endswith(":rawPredict") else f"{endpoint.rstrip('/')}:rawPredict"
    elif endpoint.startswith("http://") or endpoint.startswith("https://"):
        target_url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint.rstrip('/')}/chat/completions"
    else:
        target_url = f"http://{endpoint}:8000/v1/chat/completions"

    return target_url, model_name, None, None


def simulate_mock_stream(prompt: str, max_tokens: int = 150):
    """
    Yields synthetic SSE streaming chunks with realistic timing for mock testing.
    """
    sample_response = (
        f"This is a simulated streaming response from the /fulcrum chat engine. "
        f"I received your prompt: \"{prompt[:60]}...\". "
        f"The deployment pipeline is operating efficiently with chunked prefill and "
        f"optimized memory bandwidth allocations on Google Cloud hardware."
    )
    words = sample_response.split(" ")
    time.sleep(0.02)  # Simulate initial TTFT prefill delay

    yield {"type": "token", "content": words[0]}
    for word in words[1:]:
        time.sleep(0.005)  # Simulate decode inter-token latency
        yield {"type": "token", "content": " " + word}


# =============================================================================
# Modern Single-Page Chat Frontend (HTML, CSS, JavaScript)
# =============================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>/fulcrum chat - Model Serving Interface</title>
  <style>
    :root {
      --bg-primary: #0f141c;
      --bg-secondary: #18202c;
      --bg-card: #1f2a3a;
      --bg-input: #131a24;
      --border-color: #2b394e;
      --text-main: #f0f4f8;
      --text-muted: #94a3b8;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --bubble-user: #1e3a5f;
      --bubble-bot: #1c2636;
      --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --code-font: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-primary);
      color: var(--text-main);
      font-family: var(--font-stack);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Top Navigation Bar */
    header {
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-shrink: 0;
      z-index: 10;
    }
    .brand-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-logo {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      color: var(--accent-cyan);
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .brand-badge {
      background: rgba(59, 130, 246, 0.15);
      color: var(--accent-blue);
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .controls-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    select, input[type="text"] {
      background: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.45rem 0.75rem;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.15s ease;
    }
    select:focus, input[type="text"]:focus {
      border-color: var(--accent-blue);
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16, 185, 129, 0.12);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.25);
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 500;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
    }

    /* Main Chat Layout */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }
    #chat-history {
      flex: 1;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      scroll-behavior: smooth;
    }
    .message-row {
      display: flex;
      gap: 0.75rem;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
    }
    .message-row.user {
      justify-content: flex-end;
    }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: bold;
      flex-shrink: 0;
    }
    .avatar.user {
      background: #2563eb;
      color: white;
    }
    .avatar.bot {
      background: #0284c7;
      color: white;
    }
    .bubble {
      padding: 0.85rem 1.15rem;
      border-radius: 10px;
      line-height: 1.55;
      font-size: 0.92rem;
      position: relative;
      word-break: break-word;
      white-space: pre-wrap;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    .bubble.user {
      background-color: var(--bubble-user);
      border-bottom-right-radius: 2px;
    }
    .bubble.bot {
      background-color: var(--bubble-bot);
      border: 1px solid var(--border-color);
      border-bottom-left-radius: 2px;
      flex: 1;
    }

    /* Live Inference Stats HUD Bar */
    .stats-hud {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem 1rem;
      margin-top: 0.75rem;
      padding-top: 0.55rem;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--code-font);
    }
    .hud-stat {
      display: flex;
      align-items: center;
      gap: 0.3rem;
    }
    .hud-stat strong {
      color: var(--accent-cyan);
      font-weight: 600;
    }
    .hud-stat.tps strong {
      color: var(--accent-emerald);
    }
    .hud-stat.ttft strong {
      color: var(--accent-amber);
    }

    /* Cursor Pulse */
    .typing-cursor {
      display: inline-block;
      width: 6px;
      height: 14px;
      background-color: var(--accent-cyan);
      margin-left: 2px;
      vertical-align: middle;
      animation: blink 0.9s infinite;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }

    /* Input Dock */
    footer {
      background: var(--bg-secondary);
      border-top: 1px solid var(--border-color);
      padding: 1rem 1.5rem;
      flex-shrink: 0;
    }
    .input-container {
      max-width: 860px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .input-box-row {
      display: flex;
      gap: 0.75rem;
      align-items: flex-end;
    }
    textarea {
      flex: 1;
      background: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-size: 0.92rem;
      font-family: var(--font-stack);
      resize: none;
      outline: none;
      max-height: 160px;
      min-height: 48px;
      line-height: 1.4;
      transition: border-color 0.15s ease;
    }
    textarea:focus {
      border-color: var(--accent-blue);
    }
    .send-btn {
      background: var(--accent-blue);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0 1.25rem;
      height: 48px;
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: background-color 0.15s ease;
      flex-shrink: 0;
    }
    .send-btn:hover:not(:disabled) {
      background-color: #2563eb;
    }
    .send-btn:disabled {
      background-color: var(--border-color);
      cursor: not-allowed;
      opacity: 0.6;
    }
    .stop-btn {
      background: var(--accent-rose);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0 1rem;
      height: 48px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      display: none;
      align-items: center;
      gap: 0.4rem;
    }

    /* Generation Options Toolbar */
    .options-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--text-muted);
    }
    .param-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .param-item {
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .param-item input[type="number"] {
      width: 58px;
      background: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      padding: 0.15rem 0.35rem;
      font-size: 0.75rem;
      text-align: center;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand-group">
      <div class="brand-logo">⚖️ /fulcrum chat</div>
      <div class="brand-badge">Open Weights</div>
    </div>
    <div class="controls-group">
      <label for="deployment-select" style="font-size:0.8rem; color:var(--text-muted);">Deployment:</label>
      <select id="deployment-select">
        <option value="">Loading deployments...</option>
      </select>
      <div class="status-pill" id="status-pill">
        <div class="status-dot"></div>
        <span id="status-text">Ready</span>
      </div>
    </div>
  </header>

  <main>
    <div id="chat-history">
      <div class="message-row">
        <div class="avatar bot">⚡</div>
        <div class="bubble bot">
          Welcome to <strong>/fulcrum chat</strong>! Concurrency, chunked prefill, and GPU hardware metrics are ready.
          Select your deployment target above and begin conversing with your self-hosted model.
        </div>
      </div>
    </div>
  </main>

  <footer>
    <div class="input-container">
      <div class="input-box-row">
        <textarea id="prompt-input" rows="1" placeholder="Type your prompt... (Press Enter to send, Shift+Enter for newline)"></textarea>
        <button id="send-button" class="send-btn">Send ↵</button>
        <button id="stop-button" class="stop-btn">Stop ⏹</button>
      </div>
      <div class="options-bar">
        <div class="param-group">
          <div class="param-item">
            <span>Temp:</span>
            <input type="number" id="param-temp" min="0.0" max="2.0" step="0.1" value="0.7">
          </div>
          <div class="param-item">
            <span>Max Tokens:</span>
            <input type="number" id="param-max-tokens" min="16" max="4096" step="64" value="512">
          </div>
        </div>
        <div>Managed GCP IAM Token: <span style="color:var(--accent-emerald);">Active</span></div>
      </div>
    </div>
  </footer>

  <script>
    let activeAbortController = null;
    let chatMessages = [];

    // Load available deployments
    async function loadDeployments() {
      try {
        const resp = await fetch('/api/deployments');
        const data = await resp.json();
        const sel = document.getElementById('deployment-select');
        sel.innerHTML = '';
        if (data.deployments && data.deployments.length > 0) {
          data.deployments.forEach((d) => {
            const opt = document.createElement('option');
            opt.value = d.id;
            opt.textContent = `${d.id} (${d.model || 'model'}) [${d.platform.toUpperCase()}]`;
            sel.appendChild(opt);
          });
        } else {
          const opt = document.createElement('option');
          opt.value = "";
          opt.textContent = "Default Endpoint";
          sel.appendChild(opt);
        }
      } catch (e) {
        console.error("Failed to load deployments:", e);
      }
    }

    // Auto-resize textarea
    const promptInput = document.getElementById('prompt-input');
    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + 'px';
    });

    // Handle Enter to submit
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    document.getElementById('send-button').addEventListener('click', sendMessage);
    document.getElementById('stop-button').addEventListener('click', () => {
      if (activeAbortController) {
        activeAbortController.abort();
      }
    });

    function appendMessage(role, text) {
      const history = document.getElementById('chat-history');
      const row = document.createElement('div');
      row.className = `message-row ${role}`;

      const avatar = document.createElement('div');
      avatar.className = `avatar ${role}`;
      avatar.textContent = role === 'user' ? '👤' : '⚡';

      const bubble = document.createElement('div');
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;

      if (role === 'user') {
        row.appendChild(bubble);
        row.appendChild(avatar);
      } else {
        row.appendChild(avatar);
        row.appendChild(bubble);
      }

      history.appendChild(row);
      history.scrollTop = history.scrollHeight;
      return bubble;
    }

    async function sendMessage() {
      const text = promptInput.value.trim();
      if (!text) return;

      promptInput.value = '';
      promptInput.style.height = 'auto';

      appendMessage('user', text);
      chatMessages.push({ role: 'user', content: text });

      const botBubble = appendMessage('bot', '');
      const cursor = document.createElement('span');
      cursor.className = 'typing-cursor';
      botBubble.appendChild(cursor);

      // Create live stats container inside bubble
      const hud = document.createElement('div');
      hud.className = 'stats-hud';
      hud.innerHTML = `
        <div class="hud-stat ttft">⏱️ TTFT: <strong>--</strong></div>
        <div class="hud-stat tps">⚡ Throughput: <strong>-- tok/s</strong></div>
        <div class="hud-stat tokens">🔢 Tokens: <strong>0</strong></div>
        <div class="hud-stat total">🕒 Time: <strong>0.00s</strong></div>
      `;

      const sendBtn = document.getElementById('send-button');
      const stopBtn = document.getElementById('stop-button');
      sendBtn.style.display = 'none';
      stopBtn.style.display = 'flex';

      const depSelect = document.getElementById('deployment-select');
      const tempVal = parseFloat(document.getElementById('param-temp').value) || 0.7;
      const maxTokensVal = parseInt(document.getElementById('param-max-tokens').value) || 512;

      activeAbortController = new AbortController();

      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: chatMessages,
            deployment_id: depSelect.value,
            temperature: tempVal,
            max_tokens: maxTokensVal
          }),
          signal: activeAbortController.signal
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let assistantContent = '';

        botBubble.appendChild(hud);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop(); // Keep unfinished line in buffer

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;
            const jsonStr = trimmed.slice(6);
            if (jsonStr === '[DONE]') continue;

            try {
              const evt = JSON.parse(jsonStr);
              if (evt.type === 'token') {
                assistantContent += evt.content;
                // Update text node before cursor
                botBubble.childNodes[0].nodeValue = assistantContent;
              } else if (evt.type === 'stats') {
                hud.querySelector('.ttft strong').textContent = evt.ttft_ms ? `${evt.ttft_ms.toFixed(1)} ms` : '--';
                hud.querySelector('.tps strong').textContent = `${evt.tps.toFixed(1)} tok/s`;
                hud.querySelector('.tokens strong').textContent = evt.tokens;
                hud.querySelector('.total strong').textContent = `${evt.elapsed_s.toFixed(2)}s`;
              } else if (evt.type === 'done') {
                hud.querySelector('.ttft strong').textContent = `${evt.ttft_ms.toFixed(1)} ms`;
                hud.querySelector('.tps strong').textContent = `${evt.tps.toFixed(1)} tok/s`;
                hud.querySelector('.tokens strong').textContent = `${evt.tokens} tok`;
                hud.querySelector('.total strong').textContent = `${(evt.total_ms / 1000).toFixed(2)}s`;
                if (evt.tpot_ms) {
                  const tpotElem = document.createElement('div');
                  tpotElem.className = 'hud-stat';
                  tpotElem.innerHTML = `⏳ TPOT: <strong>${evt.tpot_ms.toFixed(1)} ms</strong>`;
                  hud.appendChild(tpotElem);
                }
              } else if (evt.type === 'error') {
                botBubble.childNodes[0].nodeValue = `[Error: ${evt.message}]`;
              }
            } catch (err) {
              console.warn("Error parsing chunk event:", err);
            }
          }
          const chatHist = document.getElementById('chat-history');
          chatHist.scrollTop = chatHist.scrollHeight;
        }

        chatMessages.push({ role: 'assistant', content: assistantContent });
      } catch (err) {
        if (err.name === 'AbortError') {
          botBubble.childNodes[0].nodeValue += " [Generation cancelled]";
        } else {
          botBubble.childNodes[0].nodeValue += ` [Error: ${err.message}]`;
        }
      } finally {
        if (cursor.parentNode) {
          cursor.remove();
        }
        sendBtn.style.display = 'flex';
        stopBtn.style.display = 'none';
        activeAbortController = null;
      }
    }

    window.addEventListener('DOMContentLoaded', loadDeployments);
  </script>
</body>
</html>
"""


class ChatRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for chat UI and SSE streaming proxy."""

    server_context: Dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server access logs to keep terminal tidy."""
        pass

    def do_GET(self) -> None:
        """Serves HTML SPA, deployments registry, and status endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        elif path == "/api/deployments":
            deployments = db.load_deployments()
            payload = json.dumps({"deployments": deployments}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif path == "/api/status":
            ctx = self.server_context
            payload = json.dumps({
                "status": "online",
                "default_model": ctx.get("model"),
                "default_endpoint": ctx.get("endpoint"),
                "mock_mode": ctx.get("mock", False),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        else:
            self.send_error(404, "Endpoint Not Found")

    def do_POST(self) -> None:
        """Handles streaming chat completion requests via Server-Sent Events."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/chat/stream":
            self.send_error(404, "Endpoint Not Found")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_data = json.loads(body)
        except Exception as e:
            self.send_error(400, f"Malformed JSON: {e}")
            return

        messages = req_data.get("messages", [])
        dep_id = req_data.get("deployment_id")
        max_tokens = req_data.get("max_tokens", 512)
        temperature = req_data.get("temperature", 0.7)

        # Prepare SSE response headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Check if server is running in mock mode
        mock_mode = self.server_context.get("mock", False)
        if mock_mode:
            last_prompt = messages[-1]["content"] if messages else "Hello"
            self._stream_mock_response(last_prompt, max_tokens)
            return

        # Target resolution
        target_url, model_name, _, _ = resolve_deployment_target(
            identifier=dep_id or self.server_context.get("deployment_id"),
            endpoint=self.server_context.get("endpoint"),
            model=self.server_context.get("model"),
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "fulcrum-chat/1.0",
        }

        # Obtain token for Vertex AI endpoints
        if "aiplatform.googleapis.com" in target_url:
            token = TOKEN_MANAGER.get_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        self._stream_remote_response(target_url, headers, payload)

    def _send_sse_event(self, event_data: Dict[str, Any]) -> None:
        """Writes an SSE chunk directly to the client connection."""
        chunk = f"data: {json.dumps(event_data)}\n\n".encode("utf-8")
        try:
            self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_mock_response(self, prompt: str, max_tokens: int) -> None:
        """Streams mock response events and computes simulated telemetry."""
        start_time = time.perf_counter()
        tokens_count = 0
        first_token_time: Optional[float] = None

        for item in simulate_mock_stream(prompt, max_tokens):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            tokens_count += 1

            self._send_sse_event(item)

            # Send periodic live stats updates every 5 tokens
            if tokens_count % 5 == 0:
                elapsed = now - start_time
                ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
                gen_time = max(0.001, now - first_token_time) if first_token_time else 0.001
                tps = tokens_count / gen_time
                self._send_sse_event({
                    "type": "stats",
                    "ttft_ms": round(ttft_ms, 1),
                    "tokens": tokens_count,
                    "tps": round(tps, 1),
                    "elapsed_s": round(elapsed, 2),
                })

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0
        decode_duration = max(0.001, end_time - (first_token_time or end_time))
        tpot_ms = (decode_duration / max(1, tokens_count - 1)) * 1000.0 if tokens_count > 1 else 0.0
        final_tps = tokens_count / decode_duration if decode_duration > 0 else 0.0

        self._send_sse_event({
            "type": "done",
            "ttft_ms": round(ttft_ms, 1),
            "tpot_ms": round(tpot_ms, 1),
            "tokens": tokens_count,
            "tps": round(final_tps, 1),
            "total_ms": round(total_time_ms, 1),
        })

    def _stream_remote_response(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        retried: bool = False,
    ) -> None:
        """Proxies streaming request to remote endpoint and measures metrics."""
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_count = 0

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                for line in response:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: ") or decoded == "data: [DONE]":
                        continue
                    chunk_str = decoded[6:]
                    try:
                        chunk = json.loads(chunk_str)
                        choices = chunk.get("choices", [])
                        if choices and "delta" in choices[0]:
                            delta = choices[0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                now = time.perf_counter()
                                if first_token_time is None:
                                    first_token_time = now
                                tokens_count += 1

                                self._send_sse_event({"type": "token", "content": content})

                                # Send live stats update every 5 tokens
                                if tokens_count % 5 == 0:
                                    elapsed = now - start_time
                                    ttft_ms = (first_token_time - start_time) * 1000.0
                                    gen_duration = max(0.001, now - first_token_time)
                                    tps = tokens_count / gen_duration
                                    self._send_sse_event({
                                        "type": "stats",
                                        "ttft_ms": round(ttft_ms, 1),
                                        "tokens": tokens_count,
                                        "tps": round(tps, 1),
                                        "elapsed_s": round(elapsed, 2),
                                    })
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000.0
            ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0
            decode_duration = max(0.001, end_time - (first_token_time or end_time))
            tpot_ms = (decode_duration / max(1, tokens_count - 1)) * 1000.0 if tokens_count > 1 else 0.0
            final_tps = tokens_count / decode_duration if decode_duration > 0 else 0.0

            self._send_sse_event({
                "type": "done",
                "ttft_ms": round(ttft_ms, 1),
                "tpot_ms": round(tpot_ms, 1),
                "tokens": tokens_count,
                "tps": round(final_tps, 1),
                "total_ms": round(total_time_ms, 1),
            })

        except urllib.error.HTTPError as e:
            # Handle token expiration (401 Unauthorized) with transparent refresh
            if e.code == 401 and not retried and "aiplatform.googleapis.com" in url:
                fresh_token = TOKEN_MANAGER.get_token(force_refresh=True)
                if fresh_token:
                    headers["Authorization"] = f"Bearer {fresh_token}"
                    return self._stream_remote_response(url, headers, payload, retried=True)

            error_msg = f"HTTP {e.code}: {e.reason}"
            try:
                err_body = e.read().decode("utf-8")
                err_json = json.loads(err_body)
                if "error" in err_json and "message" in err_json["error"]:
                    error_msg = err_json["error"]["message"]
            except Exception:
                pass
            self._send_sse_event({"type": "error", "message": error_msg})

        except Exception as e:
            self._send_sse_event({"type": "error", "message": str(e)})


def find_free_port(start_port: int = 8080) -> int:
    """Finds an available local TCP port starting from start_port."""
    for p in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum chat: Interactive Web Chat & Real-Time Telemetry Interface for deployed open-weight models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch chat web interface with the active deployment in local database
  python3 fulcrum/scripts/chat_web.py

  # Launch chat targeting a specific deployment ID
  python3 fulcrum/scripts/chat_web.py --id fulcrum-qwen3-8-27b

  # Launch chat targeting a direct Vertex AI Endpoint
  python3 fulcrum/scripts/chat_web.py --endpoint https://us-central1-aiplatform.googleapis.com/.../endpoints/123:rawPredict --model Qwen/Qwen3.8-27B

  # Run offline synthetic verification in mock mode
  python3 fulcrum/scripts/chat_web.py --mock --no-browser
""",
    )
    parser.add_argument("--id", "-i", type=str, help="Deployment ID from local deployments.json registry")
    parser.add_argument("--endpoint", "-e", type=str, help="Target API endpoint URL or host:port")
    parser.add_argument("--model", "-m", type=str, help="Model name identifier (e.g. google/gemma-3-12b-it)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind the chat web server (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--mock", action="store_true", help="Simulate synthetic token streaming for offline testing")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch the web browser")

    args = parser.parse_args()

    # Resolve target deployment information
    try:
        target_url, resolved_model, dep_id, _ = resolve_deployment_target(
            identifier=args.id,
            endpoint=args.endpoint,
            model=args.model,
        )
    except Exception as e:
        if args.mock:
            target_url = "http://mock-service/v1/chat/completions"
            resolved_model = args.model or "mock-model"
            dep_id = None
        else:
            print(f"Error resolving deployment: {e}", file=sys.stderr)
            sys.exit(1)

    port = find_free_port(args.port)
    server_address = (args.host, port)

    # Attach context to request handler
    ChatRequestHandler.server_context = {
        "deployment_id": dep_id,
        "endpoint": target_url,
        "model": resolved_model,
        "mock": args.mock,
    }

    httpd = ThreadingHTTPServer(server_address, ChatRequestHandler)
    local_url = f"http://{args.host}:{port}"

    print("=" * 72)
    print("  /fulcrum chat: Web Chat & Live Inference Telemetry Interface")
    print("=" * 72)
    print(f"Target Model:       {resolved_model}")
    print(f"Target Endpoint:    {target_url}")
    print(f"Web Interface:      {local_url}")
    if args.mock:
        print("Mode:               MOCK SIMULATION (offline testing)")
    else:
        print("Auth Management:    Automatic Google Cloud IAM OAuth2 Token Active")
    print("=" * 72)
    print("Press Ctrl+C to terminate the web chat server.\n")

    if not args.no_browser and not args.mock:
        try:
            import webbrowser
            webbrowser.open(local_url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down /fulcrum chat server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
