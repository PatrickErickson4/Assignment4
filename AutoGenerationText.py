import os
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

INPUT_PATH = "document.txt"
NB_OUTPUT_PATH = "document_nb_generated.txt"
MISTRAL_OUTPUT_PATH = "document_mistral_generated.txt"

NB_NUM_WORDS = 250        # words to generate with Naive Bayes
MISTRAL_NEW_TOKENS = 400  # tokens to generate with Mistral
NB_ALPHA = 1.0            # Laplace smoothing (not super relevant for argmax)
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# If your machine can't handle Mistral, try: "gpt2" or "distilgpt2"


# ---------------------------------------------------------
# TOKENIZATION (same schema as before)
# ---------------------------------------------------------

nltk.download("punkt", quiet=True)

def tokenize(text):
    """
    Lowercase and split into word tokens using NLTK.
    Remove tokens that are purely punctuation (no alphabetic chars).
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    return tokens


# ---------------------------------------------------------
# 1. NAIVE BAYES-STYLE AUTOREGRESSIVE GENERATION
# ---------------------------------------------------------

def build_unigram_counts(tokens):
    """
    Build a frequency counter over words.
    """
    return Counter(tokens)


def nb_greedy_next_word(counts, alpha=1.0):
    """
    Given a Counter of word counts, return the *most probable* word under
    a unigram model with Laplace smoothing.

    P(w) ∝ (count[w] + alpha) / (total + alpha * |V|)

    Since we only care about argmax, we can just maximize count[w] + alpha.
    """
    # If vocabulary is empty, bail with a dummy token
    if not counts:
        return "<unk>"

    # In practice, argmax of (count + alpha) is just argmax of count
    # But we can be explicit
    best_word = None
    best_score = -1

    for w, c in counts.items():
        score = c + alpha
        if score > best_score:
            best_score = score
            best_word = w

    return best_word


def generate_nb_words(corpus_text, num_words=NB_NUM_WORDS, alpha=NB_ALPHA):
    """
    Autoregressively generate `num_words` words using a Naive Bayes-style
    unigram model:

    - Build initial counts from the corpus.
    - For each step:
        - Pick the most probable word under the current unigram distribution.
        - Append it to the generated list.
        - Increment its count so it becomes even more probable.
    """
    tokens = tokenize(corpus_text)
    counts = build_unigram_counts(tokens)

    generated = []

    for i in range(num_words):
        next_word = nb_greedy_next_word(counts, alpha=alpha)
        generated.append(next_word)
        counts[next_word] += 1  # update frequency bank

    return generated


def run_naive_bayes_generation():
    print("=== Naive Bayes autoregressive generation ===")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file {INPUT_PATH} not found.")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    generated_words = generate_nb_words(corpus_text, num_words=NB_NUM_WORDS, alpha=NB_ALPHA)

    nb_extension = " ".join(generated_words)
    combined_text = corpus_text + "\n\n" + "[NAIVE BAYES GENERATED EXTENSION]\n\n" + nb_extension

    with open(NB_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"Naive Bayes generated {NB_NUM_WORDS} words.")
    print(f"Saved to {NB_OUTPUT_PATH}")


# ---------------------------------------------------------
# 2. MISTRAL-7B GENERATION VIA HUGGINGFACE
# ---------------------------------------------------------

def load_mistral_model_and_tokenizer(model_name=MISTRAL_MODEL_NAME):
    """
    Load Mistral (or a fallback model) via HuggingFace.
    NOTE: This requires significant GPU/CPU memory for Mistral-7B.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For big models like Mistral, fp16 + device_map="auto" is common
    # If this crashes on your machine, try a smaller model.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    return tokenizer, model


def generate_mistral_continuation(corpus_text, tokenizer, model, max_new_tokens=MISTRAL_NEW_TOKENS):
    """
    Generate a continuation of the corpus using Mistral-7B (or fallback model).
    """
    device = model.device

    prompt = corpus_text[-2000:]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def run_mistral_generation():
    print("=== Mistral-7B generation ===")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file {INPUT_PATH} not found.")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    tokenizer, model = load_mistral_model_and_tokenizer(MISTRAL_MODEL_NAME)

    generated_full = generate_mistral_continuation(
        corpus_text,
        tokenizer,
        model,
        max_new_tokens=MISTRAL_NEW_TOKENS
    )

    combined_text = corpus_text + "\n\n[MISTRAL-7B GENERATED CONTINUATION]\n\n" + generated_full

    with open(MISTRAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"Mistral generated ~{MISTRAL_NEW_TOKENS} new tokens.")
    print(f"Saved to {MISTRAL_OUTPUT_PATH}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    run_naive_bayes_generation()
    print()
    run_mistral_generation()
    print("\nDone. Inspect the two output files to compare behavior.")


if __name__ == "__main__":
    main()
