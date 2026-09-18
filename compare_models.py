from huggingface_hub import HfApi

MODEL_IDS = [
    "distilbert-base-uncased",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
]

api = HfApi()
results = []

for model_id in MODEL_IDS:
    info = api.model_info(model_id)
    if info.safetensors is not None:
        # Metadata read from the safetensors header only - no weight download.
        total_params = sum(info.safetensors.parameters.values())
    else:
        raise RuntimeError(f"No safetensors metadata available for {model_id}")
    results.append((model_id, total_params))

results.sort(key=lambda row: row[1])

name_width = max(len(name) for name, _ in results)
name_width = max(name_width, len("Model"))

header = f"{'Model':<{name_width}} | {'Parameters':>15}"
print(header)
print("-" * len(header))
for name, count in results:
    print(f"{name:<{name_width}} | {count:>15,}")
