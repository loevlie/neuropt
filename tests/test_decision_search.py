"""Tests for DecisionSearch and DecisionBackend — no real model needed.

A FakeDecisionBackend stands in for the model in-process, and a tiny stdlib
HTTP server stands in for a /v1/systemone server (H2O-Lightning-4B shape).
"""

import json
import math
import os
import random
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from neuropt import ArchSearch, Categorical, DecisionSearch, IntUniform, LogUniform, Uniform
from neuropt.arch_search import _config_key
from neuropt.backends import DecisionBackend
from neuropt.backends.base import BaseDecisionBackend, BaseLLMBackend
from neuropt.backends.decision_backend import DecisionError
from neuropt.decision_search import _distribution

SPACE = {
    "lr": LogUniform(1e-4, 1e-1),
    "dropout": Uniform(0.0, 0.5),
    "layers": IntUniform(1, 4),
    "width": IntUniform(32, 512),
    "act": Categorical(["relu", "gelu", "silu"]),
    "bn": Categorical([True, False]),
}


def train_fn(cfg):
    score = ((math.log10(cfg["lr"]) + 2.5) ** 2 + cfg["dropout"]
             + abs(cfg["layers"] - 3) + (cfg["act"] != "gelu"))
    return {"score": score, "train_losses": [2.0, 1.0], "val_losses": [2.1, 1.3]}


class FakeDecisionBackend(BaseDecisionBackend):
    """Answers every choice question; `favor(qid, keys)` picks the favored key."""

    def __init__(self, favor=None, pick=None, fail=False, fail_pick=False):
        super().__init__()
        self.favor = favor or (lambda qid, keys: keys[-1])
        self.pick = pick
        self.fail = fail
        self.fail_pick = fail_pick
        self.requests = []

    def is_available(self):
        return True

    def decide(self, state, questions):
        self.requests.append((state, questions))
        if self.fail:
            raise RuntimeError("server down")
        if "pick" in questions and self.fail_pick:
            raise RuntimeError("pick failed")
        answers = {}
        for qid, q in questions.items():
            keys = list(q["criteria"])
            if qid == "pick" and self.pick is not None:
                probs = self.pick(keys)
            else:
                fav = self.favor(qid, keys)
                probs = {k: (0.9 if k == fav else 0.1 / (len(keys) - 1)) for k in keys}
            choice = max(probs, key=probs.get)
            answers[qid] = {"type": "choice", "choice": choice,
                            "confidence": probs[choice], "probabilities": probs}
        self.total_input_tokens += 100
        return answers


@pytest.fixture
def tmp_log():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def make_search(backend, tmp_log, **kwargs):
    return DecisionSearch(train_fn, SPACE, backend=backend, log_path=tmp_log, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
#  QUESTIONS
# ══════════════════════════════════════════════════════════════════════════

class TestQuestions:
    def test_one_choice_question_per_param(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log, n_bins=8)
        params = {p.name: p for p in search._build_params([])}
        assert set(params) == set(SPACE)
        for p in params.values():
            assert p.question["type"] == "choice"
            assert list(p.question["criteria"]) == p.keys
            assert p.name in p.question["instructions"]

    def test_option_shapes(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log, n_bins=8)
        params = {p.name: p for p in search._build_params([])}
        assert len(params["lr"].keys) == 8                      # log-scale bins
        assert params["lr"].keys[0].startswith("1.00e-04 to")
        assert len(params["dropout"].keys) == 8                 # linear bins
        assert params["layers"].keys == ["1", "2", "3", "4"]    # small int range: one per value
        assert len(params["width"].keys) == 8                   # large int range: contiguous bins
        assert params["width"].options[0].lo == 32
        assert params["width"].options[-1].hi == 512
        assert params["act"].keys == ["relu", "gelu", "silu"]
        assert params["bn"].keys == ["true", "false"]

    def test_int_bins_cover_range_without_gaps(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log, n_bins=8)
        width = next(p for p in search._build_params([]) if p.name == "width")
        for a, b in zip(width.options, width.options[1:]):
            assert b.lo == a.hi + 1

    def test_history_stats_in_descriptions(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log)
        history = [
            {"id": 1, "status": "ok", "score": 0.7,
             "config": {"lr": 2e-3, "dropout": 0.1, "layers": 3, "width": 64, "act": "gelu", "bn": True}},
            {"id": 2, "status": "ok", "score": 0.4,
             "config": {"lr": 2.5e-3, "dropout": 0.2, "layers": 2, "width": 64, "act": "gelu", "bn": True}},
        ]
        search.best_config = history[1]["config"]
        search.best_score = 0.4
        params = {p.name: p for p in search._build_params(history)}
        crit = params["act"].question["criteria"]
        assert "tried 2x, best score 0.4" in crit["gelu"]
        assert "current best" in crit["gelu"]
        assert "not tried yet" in crit["relu"]
        assert "0.4" in params["act"].question["instructions"]

    def test_single_choice_param_not_asked(self, tmp_log):
        search = DecisionSearch(lambda c: 0.0, {"x": (0.0, 1.0), "opt": ["adam"]},
                                backend=FakeDecisionBackend(), log_path=tmp_log)
        params = {p.name: p for p in search._build_params([])}
        assert params["opt"].question is None
        search.run(max_evals=3)
        rows = [json.loads(line) for line in open(tmp_log)]
        assert all(r["config"]["opt"] == "adam" for r in rows)


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG GENERATION
# ══════════════════════════════════════════════════════════════════════════

class TestConfigGeneration:
    def test_greedy_candidate_follows_answers(self, tmp_log):
        # Favor the last option everywhere: top lr bin, 4 layers, silu, bn=false
        search = make_search(FakeDecisionBackend(), tmp_log, rerank=False)
        configs, source = search._get_configs([], random.Random(0))
        assert source == "decision"
        first = configs[0]
        assert first["lr"] > 10 ** (-1 - 3 / 8)
        assert first["layers"] == 4
        assert first["act"] == "silu"
        assert first["bn"] is False
        assert first["width"] >= 452

    def test_greedy_keeps_best_value_when_its_bin_is_chosen(self, tmp_log):
        best = {"lr": 0.0123, "dropout": 0.33, "layers": 2, "width": 77, "act": "relu", "bn": True}

        def favor_best(qid, keys):
            name = list(SPACE)[int(qid[1:])]
            param = next(p for p in search._build_params([]) if p.name == name)
            return keys[param.index_of(best[name])]

        search = make_search(FakeDecisionBackend(favor=favor_best), tmp_log, rerank=False)
        search.best_config = best
        search.best_score = 1.0
        history = [{"id": 1, "status": "ok", "score": 1.0, "config": dict(best)}]
        configs, _ = search._get_configs(history, random.Random(0))
        # The exact best config was already tried, so the greedy pick is a
        # duplicate and gets resampled instead.
        assert _config_key(configs[0]) != _config_key(best)
        # Without that history row, greedy keeps the best values verbatim.
        configs, _ = search._get_configs([], random.Random(0))
        assert configs[0] == pytest.approx(best)  # log10 round-trip wiggles lr

    def test_configs_valid_unique_and_logged(self, tmp_log):
        backend = FakeDecisionBackend()
        search = make_search(backend, tmp_log, batch_size=3)
        search.run(max_evals=9)
        rows = [json.loads(line) for line in open(tmp_log)]
        assert len(rows) == 9
        assert {r["source"] for r in rows} == {"decision"}
        keys = [_config_key(r["config"]) for r in rows]
        assert len(set(keys)) == 9
        for r in rows:
            cfg = r["config"]
            assert 1e-4 <= cfg["lr"] <= 1e-1
            assert 0.0 <= cfg["dropout"] <= 0.5
            assert isinstance(cfg["layers"], int) and 1 <= cfg["layers"] <= 4
            assert isinstance(cfg["width"], int) and 32 <= cfg["width"] <= 512
            assert cfg["act"] in ("relu", "gelu", "silu")
            assert cfg["bn"] in (True, False)
        assert search.llm_success == 3 and search.llm_fallback == 0
        # Two requests per iteration: per-parameter questions, then the joint pick
        assert len(backend.requests) == 6
        assert search._backend.total_input_tokens == 600

    def test_explore_zero_and_one(self, tmp_log):
        for explore in (0.0, 1.0):
            search = make_search(FakeDecisionBackend(), tmp_log, explore=explore, rerank=False)
            configs, _ = search._get_configs([], random.Random(1))
            assert len(configs) == 3

    def test_small_space_exhaustion_falls_back(self, tmp_log):
        search = DecisionSearch(lambda c: float(c["a"] == "x"), {"a": ["x", "y"]},
                                backend=FakeDecisionBackend(), log_path=tmp_log, batch_size=2)
        search.run(max_evals=4)  # only 2 distinct configs exist; must not hang
        assert search.total_experiments == 4

    def test_maximize_wording(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log, minimize=False)
        q = search._build_params([])[0].question
        assert "higher score" in q["instructions"]


# ══════════════════════════════════════════════════════════════════════════
#  JOINT PICK
# ══════════════════════════════════════════════════════════════════════════

class TestJointPick:
    def test_top_by_probability_plus_one_sampled(self, tmp_log):
        def pick(keys):
            probs = {k: 0.01 for k in keys}
            probs["c3"], probs["c5"] = 0.6, 0.3
            return probs

        search = make_search(FakeDecisionBackend(pick=pick), tmp_log, batch_size=3)
        pool = [{"lr": 1e-3, "dropout": i / 20, "layers": 1, "width": 64, "act": "relu", "bn": True}
                for i in range(8)]
        chosen = search._pick("state", pool, random.Random(0))
        assert chosen[0] is pool[2]
        assert chosen[1] is pool[4]
        assert chosen[2] not in (pool[2], pool[4])
        assert len(chosen) == 3

    def test_batch_size_one_takes_top(self, tmp_log):
        search = make_search(FakeDecisionBackend(pick=lambda keys: {"c2": 1.0}), tmp_log,
                             batch_size=1)
        pool = [{"x": i} for i in range(4)]
        assert search._pick("state", pool, random.Random(0)) == [pool[1]]

    def test_pick_failure_keeps_sampled_order(self, tmp_log):
        search = make_search(FakeDecisionBackend(fail_pick=True), tmp_log)
        pool = [{"x": i} for i in range(6)]
        assert search._pick("state", pool, random.Random(0)) == pool[:3]

    def test_rerank_off_skips_second_request(self, tmp_log):
        backend = FakeDecisionBackend()
        search = make_search(backend, tmp_log, rerank=False)
        search._get_configs([], random.Random(0))
        assert len(backend.requests) == 1


# ══════════════════════════════════════════════════════════════════════════
#  FALLBACKS AND BACKEND RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

class TestFallbacks:
    def test_backend_error_falls_back_to_random(self, tmp_log):
        search = make_search(FakeDecisionBackend(fail=True), tmp_log)
        search.run(max_evals=3)
        rows = [json.loads(line) for line in open(tmp_log)]
        assert {r["source"] for r in rows} == {"random"}
        assert search.llm_fallback == 1

    def test_backend_none_is_random_search(self, tmp_log):
        search = make_search("none", tmp_log)
        search.run(max_evals=3)
        assert search.total_experiments == 3

    def test_auto_without_server_falls_back(self, tmp_log, monkeypatch, capsys):
        monkeypatch.setenv("NEUROPT_DECISION_URL", f"http://127.0.0.1:{_closed_port()}")
        search = make_search("auto", tmp_log)
        assert search._backend is None
        assert "No decision server" in capsys.readouterr().out

    def test_rejects_llm_backend(self, tmp_log):
        class LLM(BaseLLMBackend):
            def generate(self, prompt, max_tokens=1024):
                return "[]"

            def is_available(self):
                return True

        with pytest.raises(TypeError, match="ArchSearch"):
            make_search(LLM(), tmp_log)

    def test_archsearch_rejects_decision_backend(self, tmp_log):
        with pytest.raises(TypeError, match="DecisionSearch"):
            ArchSearch(train_fn, SPACE, backend=FakeDecisionBackend(), log_path=tmp_log)
        with pytest.raises(TypeError, match="DecisionSearch"):
            ArchSearch(train_fn, SPACE, backend="decision", log_path=tmp_log)

    def test_from_sklearn_model_with_decision_backend(self, tmp_log):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        backend = FakeDecisionBackend()
        search = DecisionSearch.from_model(
            RandomForestClassifier(n_estimators=10), lambda cfg: {"score": 0.5},
            backend=backend, log_path=tmp_log)
        assert search._backend is backend  # resolved once, not rebuilt
        search.run(max_evals=3)
        assert search.total_experiments == 3


class TestDistribution:
    def test_renormalises_over_known_keys(self):
        p = _distribution(["a", "b"], {"probabilities": {"a": 0.2, "b": 0.6, "zzz": 0.2}})
        assert p == pytest.approx([0.25, 0.75])

    def test_one_hot_when_no_probabilities(self):
        assert _distribution(["a", "b"], {"choice": "b"}) == [0.0, 1.0]

    def test_bad_values_ignored(self):
        p = _distribution(["a", "b"], {"probabilities": {"a": "nan", "b": -1}, "choice": "a"})
        assert p == [1.0, 0.0]

    def test_unusable_answer_raises(self):
        with pytest.raises(ValueError):
            _distribution(["a", "b"], {"choice": "nope"})


class TestCurveDownsampling:
    def _history(self, n=500):
        return [{"id": 1, "status": "ok", "score": 1.0, "config": {"lr": 1e-3},
                 "curves": {"train_losses": [1.0 / (k + 1) for k in range(n)],
                            "val_losses": [1.1 / (k + 1) for k in range(n)]}}]

    def test_decision_state_caps_long_curves(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log)
        state = search._build_state(self._history())
        ep_lines = [line for line in state.splitlines() if line.startswith("  ep")]
        assert len(ep_lines) == 20
        assert ep_lines[0].startswith("  ep1:") and ep_lines[-1].startswith("  ep500:")

    def test_short_curves_untouched(self, tmp_log):
        search = make_search(FakeDecisionBackend(), tmp_log)
        state = search._build_state(self._history(n=5))
        assert [ln.split(":")[0] for ln in state.splitlines() if ln.startswith("  ep")] == \
            ["  ep1", "  ep2", "  ep3", "  ep4", "  ep5"]

    def test_archsearch_prompt_keeps_every_epoch(self, tmp_log):
        search = ArchSearch(train_fn, SPACE, backend="none", log_path=tmp_log)
        state = search._build_state(self._history())
        assert sum(line.startswith("  ep") for line in state.splitlines()) == 500


class TestPromptState:
    def test_state_is_prompt_without_task(self, tmp_log):
        search = ArchSearch(train_fn, SPACE, backend="none", log_path=tmp_log)
        history = [{"id": 1, "status": "ok", "score": 1.0, "config": {"lr": 1e-3},
                    "curves": {"train_losses": [2.0, 1.0], "val_losses": [2.1, 1.5]}}]
        state = search._build_state(history)
        prompt = search._build_prompt(history)
        assert prompt.startswith(state)
        assert "## Task" in prompt and "## Task" not in state
        assert "## Per-Epoch Curves" in state


# ══════════════════════════════════════════════════════════════════════════
#  HTTP BACKEND (stub /v1/systemone server)
# ══════════════════════════════════════════════════════════════════════════

def _closed_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StubServer:
    """Answers /v1/systemone like H2O-Lightning-4B's shim (uniform probabilities)."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if stub.mode == "error":
                    return self._send(500, {"error": "boom"})
                answers = {}
                for qid, q in body["questions"].items():
                    if q["type"] == "choice":
                        keys = list(q["criteria"])
                        answers[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.5,
                                        "probabilities": {k: 1 / len(keys) for k in keys}}
                    elif q["type"] == "score":
                        answers[qid] = {"type": "score", "score": 1.2, "confidence": 0.9,
                                        "legend": {"0": "low"}, "probabilities": {"0": 1.0}}
                    else:
                        answers[qid] = {"type": "noul", "noul": 0.93}
                if stub.mode == "missing":
                    answers.pop(next(iter(answers)))
                self._send(200, {"model": body.get("model"), "answers": answers,
                                 "usage": {"input_tokens": 231, "output_tokens": 3,
                                           "latency_ms": 12.5}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


QUESTIONS = {
    "lr": {"type": "choice", "instructions": "Which?", "criteria": {"low": "a", "high": "b"}},
    "urgency": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]},
    "ok": {"type": "noul", "instructions": "Is it?"},
}


class TestDecisionBackendHTTP:
    def test_round_trip(self):
        with _StubServer() as server:
            backend = DecisionBackend(base_url=server.url + "/", api_key="sk-test")
            answers = backend.decide("some state", QUESTIONS)
            assert set(answers) == set(QUESTIONS)
            assert answers["lr"]["probabilities"] == {"low": 0.5, "high": 0.5}
            assert answers["ok"]["noul"] == 0.93
            req = server.requests[0]
            assert req["path"] == "/v1/systemone"
            assert req["body"]["model"] == "h2oai/h2o-lightning-4b"
            assert req["body"]["state"] == "some state"
            assert req["headers"]["Authorization"] == "Bearer sk-test"
            backend.decide("again", QUESTIONS)
            assert backend.total_input_tokens == 462
            assert backend.total_output_tokens == 6
            assert backend.total_cost is None

    def test_no_auth_header_without_key(self, monkeypatch):
        monkeypatch.delenv("NEUROPT_DECISION_API_KEY", raising=False)
        with _StubServer() as server:
            DecisionBackend(base_url=server.url).decide("s", QUESTIONS)
            assert "Authorization" not in server.requests[0]["headers"]

    def test_env_url(self, monkeypatch):
        with _StubServer() as server:
            monkeypatch.setenv("NEUROPT_DECISION_URL", server.url)
            assert DecisionBackend().base_url == server.url

    def test_missing_answer_raises(self):
        with _StubServer(mode="missing") as server:
            with pytest.raises(DecisionError, match="no answer"):
                DecisionBackend(base_url=server.url).decide("s", QUESTIONS)

    def test_http_error_raises(self):
        with _StubServer(mode="error") as server:
            with pytest.raises(DecisionError, match="HTTP 500"):
                DecisionBackend(base_url=server.url).decide("s", QUESTIONS)

    def test_timeout_raises_decision_error(self):
        import time as _time

        class Slow(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                _time.sleep(1.0)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            backend = DecisionBackend(base_url=f"http://127.0.0.1:{httpd.server_address[1]}",
                                      timeout=0.2)
            with pytest.raises(DecisionError, match="no response"):
                backend.decide("s", QUESTIONS)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_unreachable_raises(self):
        backend = DecisionBackend(base_url=f"http://127.0.0.1:{_closed_port()}", timeout=2)
        with pytest.raises(DecisionError, match="can't reach"):
            backend.decide("s", QUESTIONS)

    def test_is_available(self):
        with _StubServer() as server:
            assert DecisionBackend(base_url=server.url).is_available()  # 404 on /health is fine
        assert not DecisionBackend(base_url=f"http://127.0.0.1:{_closed_port()}").is_available()

    def test_end_to_end_search(self, tmp_log):
        with _StubServer() as server:
            search = DecisionSearch(train_fn, SPACE, backend=DecisionBackend(base_url=server.url),
                                    log_path=tmp_log)
            search.run(max_evals=6)
            rows = [json.loads(line) for line in open(tmp_log)]
            assert {r["source"] for r in rows} == {"decision"}
            assert len(server.requests) == 4  # 2 iterations x (params + joint pick)

    def test_cli(self, tmp_log, tmp_path):
        from typer.testing import CliRunner

        from neuropt.cli import app
        script = tmp_path / "train.py"
        script.write_text('search_space = {"x": (1.0, 10.0), "act": ["a", "b"]}\n'
                          'def train_fn(config):\n    return {"score": config["x"]}\n')
        with _StubServer() as server:
            result = CliRunner().invoke(app, [
                "run", str(script), "--backend", "decision", "--decision-url", server.url,
                "-n", "3", "--log", tmp_log])
        assert result.exit_code == 0, result.output
        assert "Decision-Model-Guided Search" in result.output
        rows = [json.loads(line) for line in open(tmp_log)]
        assert {r["source"] for r in rows} == {"decision"}

    def test_cli_rejects_url_with_llm_backend(self, tmp_path):
        from typer.testing import CliRunner

        from neuropt.cli import app
        script = tmp_path / "train.py"
        script.write_text('search_space = {"x": (1.0, 10.0)}\n'
                          'def train_fn(config):\n    return {"score": config["x"]}\n')
        result = CliRunner().invoke(app, [
            "run", str(script), "--backend", "claude", "--decision-url", "http://x"])
        assert result.exit_code == 1
