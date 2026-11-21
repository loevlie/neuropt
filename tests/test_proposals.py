"""How LLM proposals are validated, and what the API backends send."""

import json
import random
from types import SimpleNamespace

import pytest

from neuropt import ArchSearch
from neuropt.backends.base import BaseLLMBackend

SPACE = {"lr": (1e-4, 1e-1), "act": ["relu", "gelu"], "batch_size": [64, 128, 256, 512],
         "use_bn": [True, False]}


class Canned(BaseLLMBackend):
    def __init__(self, responses):
        super().__init__()
        self.responses, self.calls = list(responses), []

    def generate(self, prompt, max_tokens=1024):
        self.calls.append(prompt)
        return self.responses.pop(0)

    def is_available(self):
        return True


def _search(tmp_path, backend=None, **kw):
    return ArchSearch(lambda cfg: {"score": cfg["lr"]}, SPACE, backend=backend or "none",
                      log_path=str(tmp_path / "s.jsonl"), **kw)


def _cfg(**over):
    return {"lr": 0.01, "act": "relu", "batch_size": 128, "use_bn": True, **over}


class TestSnapping:
    def test_number_between_choices_snaps_on_log_scale(self, tmp_path):
        s = _search(tmp_path)
        got = s._parse_response(json.dumps([_cfg(batch_size=96), _cfg(batch_size=192),
                                            _cfg(batch_size=5000)]))
        assert [c["batch_size"] for c in got] == [128, 256, 512]

    def test_string_case_and_spaces(self, tmp_path):
        got = _search(tmp_path)._parse_response(json.dumps([_cfg(act=" GELU ")]))
        assert got[0]["act"] == "gelu"

    def test_unknown_string_still_invalid(self, tmp_path):
        s = _search(tmp_path)
        assert s._parse_response(json.dumps([_cfg(act="swish")])) is None
        assert s.stats["invalid_configs"] == 1

    def test_numeric_choices_negative_values_use_linear_distance(self):
        from neuropt.arch_search import _nearest_choice
        assert _nearest_choice(-0.4, [-1, 0, 1]) == 0
        assert _nearest_choice(True, [0, 1]) is not True  # bools are not numbers here


class TestPartialBatches:
    def test_valid_configs_kept_when_one_is_invalid(self, tmp_path):
        s = _search(tmp_path)
        got = s._parse_response(json.dumps([_cfg(lr=0.02), _cfg(act="swish"), _cfg(lr=0.03)]))
        assert [c["lr"] for c in got] == pytest.approx([0.02, 0.03])
        assert s.stats["invalid_configs"] == 1

    def test_invalid_config_replaced_by_random_and_labelled(self, tmp_path):
        backend = Canned([json.dumps([_cfg(lr=0.02), _cfg(act="swish"), _cfg(lr=0.03)])])
        s = _search(tmp_path, backend)
        configs, sources = s._get_configs([], random.Random(0))
        assert sources == ["llm", "llm", "random"]
        assert len(configs) == 3 and s.stats["parse_failures"] == 0

    def test_run_logs_per_config_source(self, tmp_path):
        backend = Canned([json.dumps([_cfg(lr=0.02), _cfg(act="swish")])] * 2)
        s = _search(tmp_path, backend, batch_size=2)
        s.run(max_evals=2)
        rows = [json.loads(line) for line in open(tmp_path / "s.jsonl")]
        assert [r["source"] for r in rows] == ["llm", "random"]

    def test_structured_output_object_shape(self, tmp_path):
        got = _search(tmp_path)._parse_response(json.dumps({"configs": [_cfg(lr=0.05)]}))
        assert got[0]["lr"] == pytest.approx(0.05)


class TestSchema:
    def test_schema_lists_choices_and_requires_every_key(self, tmp_path):
        schema = _search(tmp_path)._response_schema()
        item = schema["properties"]["configs"]["items"]
        assert item["required"] == list(SPACE)
        assert item["additionalProperties"] is False
        assert item["properties"]["batch_size"] == {"enum": [64, 128, 256, 512]}
        assert item["properties"]["lr"]["type"] == "number"
        assert "minimum" not in json.dumps(schema)  # unsupported by structured outputs

    def test_plain_backend_gets_one_prompt(self, tmp_path):
        backend = Canned([json.dumps([_cfg()] * 3)])
        s = _search(tmp_path, backend)
        s._get_configs([], random.Random(0))
        assert "## Config Space" in backend.calls[0] and "## Task" in backend.calls[0]

    def test_backend_without_base_class_still_works(self, tmp_path):
        class Bare:  # duck-typed: only generate()
            name, total_input_tokens, total_output_tokens, total_cost = "bare", 0, 0, None

            def generate(self, prompt, max_tokens=1024):
                return json.dumps([_cfg(lr=0.07)])

            def is_available(self):
                return True

        configs, sources = _search(tmp_path, Bare())._get_configs([], random.Random(0))
        assert configs[0]["lr"] == pytest.approx(0.07) and sources[0] == "llm"


# ── API backends with fake clients ───────────────────────────────────────


class _Rejects(Exception):
    status_code = 400


class FakeAnthropic:
    def __init__(self, reject_structured=False):
        self.kwargs, self.reject = [], reject_structured
        self.messages = self

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.reject and "extra_body" in kwargs:
            raise _Rejects("output_config.format is not supported for this model")
        usage = SimpleNamespace(input_tokens=100, output_tokens=20,
                                cache_creation_input_tokens=1000, cache_read_input_tokens=0)
        return SimpleNamespace(usage=usage, content=[SimpleNamespace(type="text", text="[]")])


class TestClaudeBackend:
    def _backend(self, client, model="claude-haiku-5-5"):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend(model=model)
        b._client = client
        return b

    def test_sends_cached_system_and_schema(self):
        client = FakeAnthropic()
        b = self._backend(client)
        b.generate_structured("dynamic", system="static", schema={"type": "object"})
        kw = client.kwargs[0]
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kw["extra_body"]["output_config"]["format"]["type"] == "json_schema"
        assert kw["messages"][0]["content"] == "dynamic"
        # 100 in at $0.10, 1000 cache writes at 1.25 x $0.10, 20 out at $0.50 (per million)
        assert b.total_cost == pytest.approx((100 * 0.10 + 1000 * 0.125 + 20 * 0.50) / 1e6)

    def test_falls_back_when_structured_outputs_rejected(self, capsys):
        client = FakeAnthropic(reject_structured=True)
        b = self._backend(client)
        b.generate_structured("p", schema={"type": "object"})
        b.generate_structured("p", schema={"type": "object"})
        assert ["extra_body" in k for k in client.kwargs] == [True, False, False]
        assert "structured outputs" in capsys.readouterr().out

    def test_model_from_env(self, monkeypatch):
        from neuropt.backends.claude_backend import ClaudeBackend
        monkeypatch.setenv("NEUROPT_MODEL", "claude-sonnet-5-5")
        assert ClaudeBackend()._model == "claude-sonnet-5-5"
        monkeypatch.setenv("NEUROPT_MODEL", "gpt-4o-mini")  # not a Claude model: ignored
        assert ClaudeBackend()._model.startswith("claude-haiku")


class FakeOpenAI:
    def __init__(self, reject=False):
        self.kwargs, self.reject = [], reject
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.reject and "response_format" in kwargs:
            raise _Rejects("response_format json_schema not supported")
        msg = SimpleNamespace(content='{"configs": []}')
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=msg)])


class TestOpenAIBackend:
    def test_system_message_and_strict_schema(self):
        from neuropt.backends.openai_backend import OpenAIBackend
        b = OpenAIBackend(model="gpt-4o-mini", base_url="", api_key="x")
        b._client = FakeOpenAI()
        b.generate_structured("dyn", system="static", schema={"type": "object"})
        kw = b._client.kwargs[0]
        assert [m["role"] for m in kw["messages"]] == ["system", "user"]
        assert kw["response_format"]["json_schema"]["strict"] is True

    def test_compatible_server_without_schema_support(self, capsys):
        from neuropt.backends.openai_backend import OpenAIBackend
        b = OpenAIBackend(model="llama3", base_url="http://localhost:8000/v1")
        b._client = FakeOpenAI(reject=True)
        assert b.generate_structured("p", schema={"type": "object"}) == '{"configs": []}'
        assert "response_format" not in b._client.kwargs[-1]
        assert b.total_cost is None


class TestInitialConfigs:
    def test_evaluated_first_then_model(self, tmp_path):
        backend = Canned([json.dumps([_cfg(lr=0.02), _cfg(lr=0.03), _cfg(lr=0.04)])])
        s = _search(tmp_path, backend, initial_configs=[_cfg(lr=0.001, batch_size=100)])
        s.run(max_evals=4)
        rows = [json.loads(line) for line in open(tmp_path / "s.jsonl")]
        assert [r["source"] for r in rows] == ["initial", "llm", "llm", "llm"]
        assert rows[0]["config"]["batch_size"] == 128  # snapped into the space
        assert "#1" in backend.calls[0] or "| 1 |" in backend.calls[0]  # the model sees it

    def test_not_rerun_on_resume(self, tmp_path):
        init = [_cfg(lr=0.001)]
        _search(tmp_path, initial_configs=init).run(max_evals=2)
        s = _search(tmp_path, initial_configs=init)
        s.run(max_evals=1)
        rows = [json.loads(line) for line in open(tmp_path / "s.jsonl")]
        assert [r["source"] for r in rows] == ["initial", "random", "random"]

    def test_partial_config_allowed(self, tmp_path):
        s = _search(tmp_path, initial_configs=[{"lr": 0.5, "unknown": 1}])
        assert s.initial_configs == [{"lr": 0.1}]  # clamped, unknown key dropped

    def test_config_with_nothing_valid_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _search(tmp_path, initial_configs=[{"nope": 1}])


class TestStartFromModel:
    def test_sklearn_uses_estimator_params(self, tmp_path):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        s = ArchSearch.from_model(RandomForestClassifier(n_estimators=40, max_depth=None),
                                  lambda cfg: {"score": 0.5}, backend="none",
                                  log_path=str(tmp_path / "s.jsonl"))
        init = s.initial_configs[0]
        assert "max_depth" not in init  # None = library default, left out
        if "n_estimators" in s.search_space:
            assert init["n_estimators"] == max(s.search_space["n_estimators"].low,
                                               min(40, s.search_space["n_estimators"].high))

    def test_pytorch_uses_model_and_textbook_training(self, tmp_path):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(),
                                    torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
        s = ArchSearch.from_model(model, lambda cfg: {"score": 0.5}, backend="none",
                                  log_path=str(tmp_path / "s.jsonl"))
        init = s.initial_configs[0]
        assert init["activation"] == "gelu" and init["dropout"] == pytest.approx(0.3)
        assert (init["lr"], init["optimizer"]) == (1e-3, "adamw")

    def test_can_be_turned_off(self, tmp_path):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU())
        s = ArchSearch.from_model(model, lambda cfg: {"score": 0.5}, backend="none",
                                  log_path=str(tmp_path / "s.jsonl"), start_from_model=False)
        assert s.initial_configs == []


class TestFallbackVisibility:
    def test_summary_warns_when_many_configs_are_random(self, tmp_path, capsys):
        backend = Canned(["no json"] * 3)
        _search(tmp_path, backend).run(max_evals=6)
        out = capsys.readouterr().out
        assert "0 from the model, 6 random (100% random)" in out and "WARNING" in out

    def test_no_warning_when_model_answers(self, tmp_path, capsys):
        backend = Canned([json.dumps([_cfg(lr=x / 100), _cfg(lr=x / 100 + 0.001),
                                      _cfg(lr=x / 100 + 0.002)]) for x in range(1, 4)])
        _search(tmp_path, backend).run(max_evals=6)
        out = capsys.readouterr().out
        assert "6 from the model, 0 random" in out and "WARNING" not in out

    def test_auto_without_backend_says_random(self, tmp_path, capsys, monkeypatch):
        import neuropt.backends as backends
        monkeypatch.setattr(backends, "get_default_backend", lambda model=None: None)
        ArchSearch(lambda c: {"score": c["lr"]}, SPACE, backend="auto",
                   log_path=str(tmp_path / "s.jsonl")).run(max_evals=1)
        assert "plain random search" in capsys.readouterr().out


class TestCLIModelFlags:
    def _run(self, tmp_path, monkeypatch, *args):
        from typer.testing import CliRunner

        import neuropt.backends as backends
        from neuropt.cli import app
        seen = {}

        def fake_by_name(name, model=None):
            import os
            seen.update(name=name, model=os.environ.get("NEUROPT_MODEL"),
                        url=os.environ.get("NEUROPT_LLM_BASE_URL"))
            return None

        monkeypatch.setattr(backends, "get_backend_by_name", fake_by_name)
        monkeypatch.delenv("NEUROPT_MODEL", raising=False)
        monkeypatch.delenv("NEUROPT_LLM_BASE_URL", raising=False)
        script = tmp_path / "s.py"
        script.write_text('search_space = {"x": (1, 10)}\n'
                          'def train_fn(config): return {"score": config["x"]}\n')
        result = CliRunner().invoke(app, ["run", str(script), "-n", "1",
                                          "--log", str(tmp_path / "l.jsonl"), *args])
        return result, seen

    def test_base_url_selects_compatible_backend(self, tmp_path, monkeypatch):
        result, seen = self._run(tmp_path, monkeypatch, "--model", "qwen2.5:7b",
                                 "--base-url", "http://localhost:11434/v1")
        assert result.exit_code == 0, result.output
        assert seen == {"name": "openai-compatible", "model": "qwen2.5:7b",
                        "url": "http://localhost:11434/v1"}

    def test_model_with_named_backend(self, tmp_path, monkeypatch):
        result, seen = self._run(tmp_path, monkeypatch, "--backend", "claude",
                                 "--model", "claude-haiku-5-5")
        assert result.exit_code == 0, result.output
        assert seen["name"] == "claude" and seen["model"] == "claude-haiku-5-5"

    def test_base_url_rejected_for_claude(self, tmp_path, monkeypatch):
        result, _ = self._run(tmp_path, monkeypatch, "--backend", "claude",
                              "--base-url", "http://x/v1")
        assert result.exit_code == 1
