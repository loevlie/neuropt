"""Decision-model backend speaking the TypeSafe "System One" HTTP contract.

Experimental: used by ``DecisionSearch``, which is still being evaluated.

Jev (TypeSafe AI) and its open alternatives answer typed questions with
calibrated probabilities instead of generating text. They share one request
shape, ``POST {base_url}/v1/systemone``::

    {"model": "...", "state": "<evidence>",
     "questions": {"q1": {"type": "choice", "instructions": "...",
                          "criteria": {"a": "description", "b": "..."}}}}

and answer with ``{"answers": {"q1": {"choice": "a", "probabilities": {...}}},
"usage": {"input_tokens": N, ...}}``.

The default target is H2O-Lightning-4B (``h2oai/h2o-lightning-4b``,
Apache-2.0), the strongest open-weight model on the independent JevBench
v1.6.1 composite score. Its ``serve.sh`` listens on ``127.0.0.1:8741``. Any
other compatible server works by changing ``base_url``: ``jevk5-serve``, Kev,
decisio, or hosted Jev via OpenRouter (``https://openrouter.ai/api``).
"""

import json
import os
import urllib.error
import urllib.request

from neuropt.backends.base import BaseDecisionBackend

DEFAULT_URL = "http://127.0.0.1:8741"
DEFAULT_MODEL = "h2oai/h2o-lightning-4b"
ENDPOINT = "/v1/systemone"


class DecisionError(RuntimeError):
    """The decision server failed or returned something unusable."""


class DecisionBackend(BaseDecisionBackend):
    """Client for a ``/v1/systemone`` decision server.

    Args:
        base_url: Server root. Defaults to ``$NEUROPT_DECISION_URL``, then
            ``http://127.0.0.1:8741`` (H2O-Lightning-4B's ``serve.sh``).
        model: Model id sent with each request. Self-hosted servers mostly
            echo it; hosted APIs (e.g. ``typesafe/jev-1.14`` on OpenRouter)
            use it to route.
        api_key: Bearer token. Defaults to ``$NEUROPT_DECISION_API_KEY``.
            Only needed for hosted APIs.
        timeout: Seconds to wait for one request. Every question re-reads
            the state, so a request with many questions on a local
            (CPU/Apple) server can take minutes.
    """

    def __init__(self, base_url: str | None = None, model: str | None = DEFAULT_MODEL,
                 api_key: str | None = None, timeout: float = 300.0):
        super().__init__()
        url = base_url or os.environ.get("NEUROPT_DECISION_URL") or DEFAULT_URL
        self.base_url = url.rstrip("/")
        self.model = model
        self._api_key = api_key or os.environ.get("NEUROPT_DECISION_API_KEY")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"DecisionBackend({self.model or 'default'} @ {self.base_url})"

    def _headers(self):
        headers = {"Content-Type": "application/json", "User-Agent": "neuropt"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def decide(self, state: str, questions: dict) -> dict:
        body = {"state": state, "questions": questions}
        if self.model:
            body["model"] = self.model
        url = self.base_url + ENDPOINT
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise DecisionError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise DecisionError(f"can't reach decision server at {url}: {e.reason}") from e
        except TimeoutError as e:
            raise DecisionError(f"no response from {url} within {self.timeout:g}s") from e
        except json.JSONDecodeError as e:
            raise DecisionError(f"non-JSON response from {url}") from e

        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, dict):
            raise DecisionError(f"response from {url} has no 'answers' object")
        missing = [qid for qid in questions if qid not in answers]
        if missing:
            raise DecisionError(f"no answer for question(s) {missing}")

        usage = payload.get("usage") or {}
        self.total_input_tokens += _as_int(usage.get("input_tokens"))
        self.total_output_tokens += _as_int(usage.get("output_tokens"))
        return {qid: answers[qid] for qid in questions}

    def is_available(self) -> bool:
        """True if something answers HTTP at ``base_url`` (any status code)."""
        request = urllib.request.Request(self.base_url + "/health", headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=2):
                return True
        except urllib.error.HTTPError:
            return True  # the server is up, it just has no /health route
        except (urllib.error.URLError, OSError, ValueError):
            return False


def _as_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
