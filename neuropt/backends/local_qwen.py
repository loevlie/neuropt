"""Local Qwen2.5-1.5B-Instruct backend running on CPU (deprecated).

Use the openai-compatible backend with a local server (Ollama, vLLM) instead:
it is faster, runs any model, and supports structured outputs.
"""

from neuropt.backends.base import BaseLLMBackend


class QwenBackend(BaseLLMBackend):
    """Qwen2.5-1.5B-Instruct via transformers, pinned to CPU.

    Lazy-loads on first call. Runs on CPU to avoid GPU contention with training.
    """

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self):
        super().__init__()
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[QwenBackend] Loading {self.MODEL_ID} on CPU...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True)
        kwargs = dict(device_map="cpu", trust_remote_code=True)
        try:  # transformers >= 4.56 calls it dtype; older versions torch_dtype
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, dtype=torch.float32, **kwargs)
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, torch_dtype=torch.float32, **kwargs)
        self._model.eval()
        print("[QwenBackend] Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import torch
        self._load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=True, temperature=0.7, top_p=0.9)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        self.total_input_tokens += inputs["input_ids"].shape[1]
        self.total_output_tokens += generated.shape[0]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
