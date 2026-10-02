"""Models under test. Key -> (HF id, is_instruct).

Training-data cutoffs (checked 2026-10-02):
  Llama 3.1 8B     Dec 2023 (official, model card); released 2024-07-23
  Qwen2.5 7B       not officially stated; released 2024-09-19
  Mistral 7B v0.3  not officially stated; v0.3 = v0.2 + extended vocab;
                   v0.2 base released 2024-03-23
Control text for contamination checks should postdate the latest release
(2024-09-19) so it is unseen by every model.
"""

MODELS = {
    "qwen-base": ("Qwen/Qwen2.5-7B", False),
    "qwen-instruct": ("Qwen/Qwen2.5-7B-Instruct", True),
    "llama-base": ("meta-llama/Llama-3.1-8B", False),
    "llama-instruct": ("meta-llama/Llama-3.1-8B-Instruct", True),
    "mistral-base": ("mistralai/Mistral-7B-v0.3", False),
    "mistral-instruct": ("mistralai/Mistral-7B-Instruct-v0.3", True),
}
