from collections import defaultdict

from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")

component_counts = defaultdict(int)

for name, param in model.named_parameters():
    parts = name.split(".")
    if parts[0] == "transformer" and parts[1] == "layer":
        component = f"transformer.layer.{parts[2]}"
    else:
        component = parts[0]
    component_counts[component] += param.numel()

total_params = sum(component_counts.values())

rows = sorted(component_counts.items(), key=lambda kv: kv[1], reverse=True)

name_width = max(len(name) for name, _ in rows)
name_width = max(name_width, len("Component"))

header = f"{'Component':<{name_width}} | {'Parameters':>12} | {'% of Total':>10}"
print(header)
print("-" * len(header))
for name, count in rows:
    pct = 100 * count / total_params
    print(f"{name:<{name_width}} | {count:>12,} | {pct:>9.2f}%")

print("-" * len(header))
print(f"{'TOTAL':<{name_width}} | {total_params:>12,} | {100.0:>9.2f}%")
