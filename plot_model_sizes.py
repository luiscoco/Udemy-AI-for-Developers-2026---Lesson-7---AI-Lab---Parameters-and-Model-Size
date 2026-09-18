import matplotlib.pyplot as plt
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
    if info.safetensors is None:
        raise RuntimeError(f"No safetensors metadata available for {model_id}")
    total_params = sum(info.safetensors.parameters.values())
    results.append((model_id, total_params))

results.sort(key=lambda row: row[1])
names = [name for name, _ in results]
counts = [count for _, count in results]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(names, counts, color="#4C72B0")
ax.set_yscale("log")
ax.set_ylabel("Parameter count (log scale)")
ax.set_title("Model Size Comparison: Millions to Billions of Parameters")

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.05,
        f"{count:,}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig("model_sizes_chart.png", dpi=150)
print("Saved model_sizes_chart.png")
