"""Curated Hugging Face checkpoints, smallest first, 70B ceiling.

Only public / standard-transformer repos are listed so that the generic
layer resolver in ``openllm.runner`` can find ``model.layers`` (or the
equivalent) without architecture-specific code.  Entries marked ``gated``
need ``HF_TOKEN`` set to a token that has accepted the licence.
"""

from dataclasses import dataclass

MAX_PARAMS_B = 70.0


@dataclass(frozen=True)
class Entry:
    key: str
    repo: str
    label: str
    params_b: float
    approx_gb: float
    gated: bool = False

    @property
    def fits_limit(self):
        return self.params_b <= MAX_PARAMS_B


CATALOG = (
    Entry("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2 135M", 0.14, 0.3),
    Entry("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B", 0.5, 1.0),
    Entry("llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B", 1.2, 2.5, True),
    Entry("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B", 1.5, 3.1),
    Entry("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B", 1.7, 3.4),
    Entry("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B", 3.1, 6.2),
    Entry("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B", 3.2, 6.4, True),
    Entry("phi3.5-mini", "microsoft/Phi-3.5-mini-instruct", "Phi-3.5 mini 3.8B", 3.8, 7.7),
    Entry("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B", 7.6, 15.2),
    Entry("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B v0.3", 7.2, 14.5),
    Entry("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B", 8.0, 16.1, True),
    Entry("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", "Qwen2.5 14B", 14.7, 29.5),
    Entry("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "Qwen2.5 32B", 32.5, 65.0),
    Entry("llama3.3-70b", "meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B", 70.6, 141.0, True),
)

BY_KEY = {entry.key: entry for entry in CATALOG}
BY_REPO = {entry.repo: entry for entry in CATALOG}


def lookup(name):
    """Resolve a catalog key or a raw ``owner/repo`` id to an Entry."""
    name = (name or "").strip()
    if name in BY_KEY:
        return BY_KEY[name]
    if name in BY_REPO:
        return BY_REPO[name]
    if "/" in name:
        slug = name.replace("/", "__")
        return Entry(slug, name, name, 0.0, 0.0)
    raise KeyError("unknown model %r (try: openllm list)" % name)


def slug_for(repo):
    return repo.replace("/", "__")
