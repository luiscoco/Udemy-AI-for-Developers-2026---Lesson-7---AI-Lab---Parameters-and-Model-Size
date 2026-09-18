# Lesson 7 AI Lab: Parameters and Model Size — Walkthrough

This lab explores what "model parameters" actually mean in practice: how to count them, where they live inside a model, how model size compares across different architectures, and whether a bigger model is automatically a better one. Each section below corresponds to one script in this folder.

## Setup

Before running any of the scripts, we set up an isolated Python environment so the lab's dependencies don't clash with anything else on the machine:

1. Checked the Python version (3.10+ required — we had 3.14.0).
2. Created a `requirements.txt` listing the four libraries the lab needs: `torch`, `transformers`, `huggingface_hub`, `matplotlib`.
3. Created a virtual environment (`.venv`) and installed the requirements into it.
4. Ran a tiny `test_imports.py` script that just imports all four libraries and prints their versions, to confirm the install worked before writing any real code.

**Why a virtual environment?** It keeps this lab's package versions separate from any other Python project on your machine, so nothing you install here can break another project (or vice versa).

---

## Exercise 1 — `count_params.py`: How many parameters does a real model have?

**Goal:** Load an actual pretrained model and count its parameters directly, instead of just reading a number off a website.

**Steps:**
1. Use `transformers.AutoModel.from_pretrained("distilbert-base-uncased")` to download and load DistilBERT's weights.
2. Every PyTorch model is made of `nn.Parameter` tensors (weight matrices and bias vectors). We loop over `model.parameters()` and sum up `p.numel()` (number of elements) for each tensor to get the **total** parameter count.
3. We do the same sum but filtered to `p.requires_grad` to get the **trainable** parameter count (parameters PyTorch will update during training — as opposed to ones that might be frozen).
4. Format both numbers with commas (Python's `f"{n:,}"`) so a number like `66362880` reads as `66,362,880`.

**Result:** DistilBERT has **66,362,880** total parameters, and all of them are trainable (nothing is frozen by default).

**Key takeaway:** A "parameter count" isn't a marketing number — it's literally the sum of the sizes of every weight matrix and bias vector in the network. You can compute it yourself with two lines of PyTorch.

---

## Exercise 2 — `layer_breakdown.py`: Where do those parameters actually live?

**Goal:** Instead of one big total, break the parameter count down by component, to see which parts of the model are "expensive."

**Steps:**
1. Load the same DistilBERT model.
2. Loop over `model.named_parameters()`, which gives you both the tensor **and** its dotted name (e.g. `transformer.layer.3.attention.q_lin.weight`).
3. Group each parameter by its top-level component: the `embeddings` table, or one of the six `transformer.layer.N` blocks.
4. Sum parameter counts per group, compute each group's percentage of the total, and print a sorted table (largest component first).

**Result:**

| Component | Parameters | % of Total |
|---|---|---|
| embeddings | 23,835,648 | 35.92% |
| transformer.layer.0–5 | 7,087,872 each | 10.68% each |

**Key takeaway:** The embedding table (which maps every vocabulary token to a vector) alone accounts for over a third of the entire model — more than any single transformer layer. This is a common pattern in smaller language models: vocabulary size has an outsized effect on parameter count.

---

## Exercise 3 — `compare_models.py`: Comparing sizes across different models without downloading gigabytes of weights

**Goal:** Compare parameter counts across four different models (DistilBERT, GPT-2, GPT-2-medium, GPT-2-large) *without* downloading their full weight files just to count numbers.

**Steps:**
1. Use `huggingface_hub.HfApi().model_info(model_id)` to query the Hugging Face Hub's metadata about each model.
2. Models stored in the `safetensors` format ship a small header describing every tensor's shape and dtype. `model_info(...).safetensors.parameters` gives you the total parameter count straight from that header — no need to download and load gigabytes of actual weight values.
3. Sort the results smallest to largest and print a simple table.

**Result:**

| Model | Parameters |
|---|---|
| distilbert-base-uncased | 66,985,530 |
| gpt2 | 137,022,720 |
| gpt2-medium | 379,988,992 |
| gpt2-large | 811,778,816 |

**Key takeaway:** Model registries often expose lightweight metadata (like a safetensors header) that lets you answer "how big is this model?" cheaply. You don't need to load a model into memory just to know its size.

*Small note:* DistilBERT's count here (66,985,530) is slightly different from Exercise 1's (66,362,880) because this version includes the full checkpoint — including a masked-language-modeling head — while `AutoModel` in Exercise 1 loads only the base encoder.

---

## Exercise 4 — `tiny_net.py`: Watching parameters actually change during training

**Goal:** All the previous exercises treated parameters as static numbers. This exercise makes it concrete by showing the *actual values* change as a tiny network learns.

**Steps:**
1. Define a minimal 2-layer network in PyTorch: `Linear(4 → 8)` then `Linear(8 → 1)`, with a ReLU in between.
2. Print the total parameter count (49 — small enough to reason about by hand: 4×8 weights + 8 biases in layer 1, plus 8×1 weights + 1 bias in layer 2).
3. Print `model.layer1.weight.data` — the raw numbers — **before** any training.
4. Generate a toy regression task: random 4-dimensional inputs `X`, and a target `y` equal to the sum of each input's four values (plus a little noise).
5. Train for 20 steps using plain SGD and mean-squared-error loss.
6. Print `model.layer1.weight.data` again **after** training and compare.

**Result:** Loss dropped steadily from 3.71 to 3.14 over 20 steps, and the layer-1 weight values visibly shifted between the "before" and "after" printouts (e.g. one row went from `[-0.4777, -0.3311, -0.2061, 0.0185]` to `[-0.5304, -0.3514, -0.2270, 0.0017]`).

**Key takeaway:** "Training a model" literally means: compute a loss, then nudge every parameter tensor a little bit in the direction that reduces that loss. There's no magic — the weight tensors you can print and inspect are the same numbers `optimizer.step()` is updating.

---

## Exercise 5 — `compare_performance.py`: Bigger isn't automatically better

**Goal:** Test whether a larger, general-purpose model beats a smaller model that's been fine-tuned for a specific task.

**Steps:**
1. Load `distilbert-base-uncased-finetuned-sst-2-english` (66M-ish parameters, fine-tuned specifically for positive/negative sentiment classification) via the `sentiment-analysis` pipeline, which returns a clean label and confidence score.
2. Load plain `gpt2` (137M parameters, a general-purpose text generator with no sentiment fine-tuning).
3. Run both models on the same 5 sentences with obvious positive/negative sentiment.
4. For GPT-2, since it can't return a structured label, we **prompt** it (`Review: "..."\nSentiment (positive or negative):`) and let it generate a few tokens, then search the generated text for the words "positive" or "negative."
5. Print both sets of results side by side and score accuracy against the expected labels.

**Result:** DistilBERT scored 5/5 with 1.00 confidence on every sentence. GPT-2 scored 0/5 — not because it "got the sentiment wrong," but because it kept generating a plausible continuation of the sentence (e.g. `"I'm so glad`) instead of literally outputting the word "positive," so our simple keyword-matching parser couldn't extract a label from it.

**Key takeaway:** Parameter count isn't the whole story. A 66M-parameter model *fine-tuned* for a specific task beat a 137M-parameter general-purpose model at that exact task — because fine-tuning teaches a model the specific input/output format you need, while a general model has to be coaxed into the right format via prompting, and that's fragile. Bigger models shine when you need broad, flexible capability; smaller fine-tuned models shine when you have one job and want it done reliably and cheaply.

---

## Exercise 6 — `plot_model_sizes.py`: Visualizing scale differences

**Goal:** Parameter counts spanning 67 million to 812 million are hard to compare on a normal chart — the smallest bar would be nearly invisible next to the largest. A **log-scale** y-axis fixes this.

**Steps:**
1. Reuse the same Hugging Face Hub metadata lookup from Exercise 3 to gather parameter counts for all four models.
2. Sort them smallest to largest.
3. Plot a bar chart with `matplotlib`, setting `ax.set_yscale("log")` so equal *visual* steps up the y-axis represent equal *multiplicative* jumps in parameter count, not equal additive jumps.
4. Label each bar with its exact parameter count and add the title "Model Size Comparison: Millions to Billions of Parameters."
5. Save the figure to `model_sizes_chart.png`.

**Result:** A four-bar chart clearly showing DistilBERT as the smallest, then GPT-2, GPT-2-medium, and GPT-2-large each roughly doubling to tripling in size — readable at a glance thanks to the log scale.

**Key takeaway:** When comparing quantities that span orders of magnitude (thousands vs. millions vs. billions), a log scale keeps every value visually legible on the same chart, instead of the smaller bars getting crushed down to nothing next to the largest one.

---

## Files in this folder

| File | Purpose |
|---|---|
| `requirements.txt` | Lab dependencies |
| `test_imports.py` | Sanity-check that all libraries import correctly |
| `count_params.py` | Exercise 1: total/trainable parameter count |
| `layer_breakdown.py` | Exercise 2: parameter count by component |
| `compare_models.py` | Exercise 3: parameter counts across 4 models |
| `tiny_net.py` | Exercise 4: watching weights change during training |
| `compare_performance.py` | Exercise 5: fine-tuned small model vs. general large model |
| `plot_model_sizes.py` | Exercise 6: log-scale bar chart of model sizes |
| `model_sizes_chart.png` | Output image from Exercise 6 |
