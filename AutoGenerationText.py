import os
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
INPUT_PATH = "document.txt"
NB_OUTPUT_PATH = "document_nb_generated.txt"
MISTRAL_OUTPUT_PATH = "document_mistral_generated.txt"
NB_NUM_WORDS = 250        
MISTRAL_NEW_TOKENS = 400  
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
NB_ALPHA = 1


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


def build_unigram_counts(tokens):
    return Counter(tokens)


def nb_greedy_next_word(counts, alpha=1.0):

    if not counts:
        return "<unk>"


    best_word = None
    best_score = -1

    for w, c in counts.items():
        score = c + alpha
        if score > best_score:
            best_score = score
            best_word = w

    return best_word


def generate_nb_words(corpus_text, num_words=NB_NUM_WORDS, alpha=NB_ALPHA):

    tokens = tokenize(corpus_text)
    counts = build_unigram_counts(tokens)

    generated = []

    for i in range(num_words):
        next_word = nb_greedy_next_word(counts, alpha=alpha)
        generated.append(next_word)

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



def load_mistral_model_and_tokenizer(model_name=MISTRAL_MODEL_NAME):

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    return tokenizer, model


def generate_mistral_continuation(corpus_text, tokenizer, model, max_new_tokens=MISTRAL_NEW_TOKENS):

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


def main():
    run_naive_bayes_generation()
    print()
    run_mistral_generation()
    print("\nDone. Inspect the two output files to compare behavior.")


if __name__ == "__main__":
    main()
