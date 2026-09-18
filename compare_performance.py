from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

SENTENCES = [
    ("I absolutely loved this movie, it was fantastic!", "positive"),
    ("This was the worst service I have ever experienced.", "negative"),
    ("The food was delicious and the staff were so friendly.", "positive"),
    ("I'm so disappointed, the product broke after one day.", "negative"),
    ("What a wonderful, uplifting performance by the whole cast.", "positive"),
]

# --- DistilBERT (fine-tuned for sentiment) ---
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

distilbert_results = []
for text, expected in SENTENCES:
    result = sentiment_pipe(text)[0]
    label = result["label"].lower()
    distilbert_results.append((label, result["score"]))

# --- GPT-2 (general-purpose, prompted to classify) ---
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_model.eval()

def gpt2_classify(text):
    prompt = (
        f'Review: "{text}"\n'
        "Sentiment (positive or negative):"
    )
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = gpt2_model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    generated = gpt2_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    generated_lower = generated.lower()
    if "positive" in generated_lower:
        parsed = "positive"
    elif "negative" in generated_lower:
        parsed = "negative"
    else:
        parsed = f"unclear ({generated.strip()!r})"
    return parsed

gpt2_results = [gpt2_classify(text) for text, _ in SENTENCES]

# --- Print side by side ---
col1, col2, col3, col4 = 45, 10, 22, 10
print(f"{'Sentence':<{col1}} | {'Expected':<{col2}} | {'DistilBERT':<{col3}} | {'GPT-2':<{col4}}")
print("-" * (col1 + col2 + col3 + col4 + 9))

distilbert_correct = 0
gpt2_correct = 0

for (text, expected), (db_label, db_score), gpt2_label in zip(
    SENTENCES, distilbert_results, gpt2_results
):
    display_text = text if len(text) <= col1 else text[: col1 - 3] + "..."
    db_display = f"{db_label} ({db_score:.2f})"
    print(f"{display_text:<{col1}} | {expected:<{col2}} | {db_display:<{col3}} | {gpt2_label:<{col4}}")

    if db_label == expected:
        distilbert_correct += 1
    if gpt2_label == expected:
        gpt2_correct += 1

print()
print(f"DistilBERT accuracy: {distilbert_correct}/{len(SENTENCES)}")
print(f"GPT-2 accuracy:      {gpt2_correct}/{len(SENTENCES)}")
print()
print(
    "Note: DistilBERT was fine-tuned specifically for sentiment classification "
    "(SST-2), so it produces a clean, confident label directly. GPT-2 is a "
    "general-purpose text generator with no sentiment fine-tuning, so it has to "
    "be prompted and its free-text output parsed - it is more likely to produce "
    "unclear or off-format responses, making it less reliable for this specific "
    "task even though it is a larger model."
)
