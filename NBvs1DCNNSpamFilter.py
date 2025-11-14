import os
from collections import Counter
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

# ------------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------------

# Path to your CSV from the Kaggle dataset
CSV_PATH = "emails.csv"  # adjust if needed

MAX_LEN = 60      # "60 chunks" = 60 word time steps per email
MIN_FREQ = 1      # min frequency to keep a word in vocab
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
NB_ALPHA = 1.0    # Laplace smoothing for Naive Bayes


# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: "text" and "spam" (0 = ham, 1 = spam)
    texts = df["text"].astype(str).tolist()
    labels = df["spam"].astype(int).values
    return texts, labels


# ------------------------------------------------------------------
# 2. TOKENIZATION & VOCAB
# ------------------------------------------------------------------

nltk.download("punkt", quiet=True)

def tokenize(text):
    """
    Lowercase and split into word tokens.
    Filter out tokens that are purely punctuation.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    return tokens


def build_vocab(texts, min_freq=MIN_FREQ):
    counter = Counter()
    for txt in texts:
        counter.update(tokenize(txt))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]


def pad_sequence(seq, max_len=MAX_LEN, pad_idx=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


# ------------------------------------------------------------------
# 3. NAIVE BAYES (word-frequency, no TF-IDF)
# ------------------------------------------------------------------

class NaiveBayesWordFreq:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_word_counts = {}
        self.class_totals = {}
        self.class_priors = {}
        self.vocab = set()

    def fit(self, texts, labels):
        # labels are 0 (ham), 1 (spam)
        self.class_word_counts = {0: Counter(), 1: Counter()}
        class_counts = Counter(labels)
        self.vocab = set()

        for txt, y in zip(texts, labels):
            words = tokenize(txt)
            self.class_word_counts[y].update(words)
            self.vocab.update(words)

        # Total word counts per class
        self.class_totals = {
            c: sum(self.class_word_counts[c].values()) for c in [0, 1]
        }

        # Priors P(c)
        total_docs = len(labels)
        self.class_priors = {
            c: class_counts[c] / total_docs for c in [0, 1]
        }

    def predict_log_proba(self, text):
        words = tokenize(text)
        V = len(self.vocab)
        log_probs = {}

        for c in [0, 1]:
            # log P(c)
            log_prob = np.log(self.class_priors[c] + 1e-12)

            total_words_c = self.class_totals[c]
            for w in words:
                count_wc = self.class_word_counts[c][w]
                # P(w|c) with Laplace smoothing
                pwc = (count_wc + self.alpha) / (total_words_c + self.alpha * V)
                log_prob += np.log(pwc + 1e-12)

            log_probs[c] = log_prob

        return log_probs

    def predict_proba(self, text):
        log_probs = self.predict_log_proba(text)
        # Convert log probs to normalized probabilities
        max_log = max(log_probs.values())
        exps = {c: np.exp(lp - max_log) for c, lp in log_probs.items()}
        Z = sum(exps.values())
        return {c: v / Z for c, v in exps.items()}

    def predict(self, text):
        log_probs = self.predict_log_proba(text)
        return max(log_probs, key=log_probs.get)


# ------------------------------------------------------------------
# 4. DATASET FOR CNN
# ------------------------------------------------------------------

class EmailDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.vocab = vocab
        self.texts = texts
        self.labels = labels
        self.pad_idx = vocab["<pad>"]

        self.sequences = [
            pad_sequence(encode(t, self.vocab), MAX_LEN, self.pad_idx)
            for t in self.texts
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ------------------------------------------------------------------
# 5. 1D CNN TEXT CLASSIFIER (TextCNN-style)
# ------------------------------------------------------------------

class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # One conv layer per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)          # (B, T, E)
        emb = emb.transpose(1, 2)        # (B, E, T) for Conv1d

        conv_outs = []
        for conv in self.convs:
            # conv(emb): (B, num_filters, T-k+1)
            c = conv(emb)
            # global max pooling over time dimension
            c = torch.relu(c)
            c, _ = torch.max(c, dim=2)   # (B, num_filters)
            conv_outs.append(c)

        h = torch.cat(conv_outs, dim=1)  # (B, num_filters * len(kernel_sizes))
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)   # (B,)
        return logits


def train_cnn(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # Simple validation accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}, "
              f"train acc: {train_acc:.4f}, val acc: {val_acc:.4f}")

    return model


def predict_cnn(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


# ------------------------------------------------------------------
# 6. PLOTTING HELPERS
# ------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, title, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title
    )

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

def plot_combined_roc(
    fpr_nb, tpr_nb, auc_nb,
    fpr_cnn, tpr_cnn, auc_cnn,
    filename
):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Naive Bayes ROC
    ax.plot(
        fpr_nb, tpr_nb,
        label=f"Naive Bayes (AUC = {auc_nb:.3f})",
        linewidth=2
    )

    # CNN ROC
    ax.plot(
        fpr_cnn, tpr_cnn,
        label=f"TextCNN (AUC = {auc_cnn:.3f})",
        linewidth=2
    )

    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', label="Random Baseline")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Naive Bayes vs TextCNN")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_accuracy_table(acc_nb, acc_cnn, filename):
    fig, ax = plt.subplots()
    ax.axis("off")
    table_data = [
        ["Model", "Accuracy"],
        ["Naive Bayes", f"{acc_nb:.3f}"],
        ["TextCNN", f"{acc_cnn:.3f}"],
    ]
    table = ax.table(cellText=table_data, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------------

def main():
    out_dir = os.getcwd()
    print(f"Working directory: {out_dir}")

    # 1. Load data
    texts, labels = load_dataset(CSV_PATH)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # ------------------------------------------------------------------
    # Naive Bayes: word-frequency baseline
    # ------------------------------------------------------------------
    nb = NaiveBayesWordFreq(alpha=NB_ALPHA)
    nb.fit(X_train, y_train)

    y_pred_nb = []
    y_prob_nb = []

    for txt in X_test:
        probs = nb.predict_proba(txt)
        y_prob_nb.append(probs[1])  # probability of spam class (1)
        y_pred_nb.append(1 if probs[1] >= 0.5 else 0)

    y_prob_nb = np.array(y_prob_nb)
    y_pred_nb = np.array(y_pred_nb)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    auc_nb = roc_auc_score(y_test, y_prob_nb)

    print(f"Naive Bayes accuracy: {acc_nb:.4f}, AUC: {auc_nb:.4f}")

    plot_confusion_matrix(
        cm_nb,
        classes=["ham", "spam"],
        title="Naive Bayes Confusion Matrix",
        filename=os.path.join(out_dir, "nb_confusion_matrix.png")
    )

    # ------------------------------------------------------------------
    # TextCNN: 1D CNN over word sequences
    # ------------------------------------------------------------------
    vocab = build_vocab(X_train, min_freq=MIN_FREQ)
    print(f"Vocab size (including <pad>, <unk>): {len(vocab)}")

    # Further split train into train/val for CNN
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    train_ds = EmailDataset(X_tr, y_tr, vocab)
    val_ds = EmailDataset(X_val, y_val, vocab)
    test_ds = EmailDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TextCNNClassifier(vocab_size=len(vocab)).to(device)
    model = train_cnn(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR)

    y_true_cnn, y_pred_cnn, y_prob_cnn = predict_cnn(model, test_loader, device)

    acc_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
    cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_prob_cnn)
    auc_cnn = roc_auc_score(y_true_cnn, y_prob_cnn)

    print(f"TextCNN accuracy: {acc_cnn:.4f}, AUC: {auc_cnn:.4f}")

    plot_confusion_matrix(
        cm_cnn,
        classes=["ham", "spam"],
        title="TextCNN Confusion Matrix",
        filename=os.path.join(out_dir, "cnn_confusion_matrix.png")
    )

    plot_combined_roc(
        fpr_nb, tpr_nb, auc_nb,
        fpr_cnn, tpr_cnn, auc_cnn,
        filename=os.path.join(out_dir, "combined_roc_curve.png")
    )

    # ------------------------------------------------------------------
    # Accuracy table comparing the two models
    # ------------------------------------------------------------------
    save_accuracy_table(
        acc_nb,
        acc_cnn,
        filename=os.path.join(out_dir, "accuracy_table.png")
    )

    print("Saved:")
    print("  nb_confusion_matrix.png")
    print("  nb_roc_curve.png")
    print("  cnn_confusion_matrix.png")
    print("  cnn_roc_curve.png")
    print("  accuracy_table.png")


if __name__ == "__main__":
    main()
