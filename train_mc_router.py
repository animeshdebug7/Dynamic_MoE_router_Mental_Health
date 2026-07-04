# Trains the MLP baseline router on Llama hidden-state embeddings of the source
# PDFs (subsampled to 3k/class, class-weighted loss). Caches embeddings so this
# runs once. Used only as the baseline the zero-shot router is compared against.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import PyPDF2

INPUT_DIM = 2048
HIDDEN_DIM = 256
NUM_EXPERTS = 5
DROPOUT_RATE = 0.3
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
EMBED_BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_PER_CLASS = 3000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASE_MODEL_PATH = "/Users/animeshsingh/Desktop/Models/Llama 3.2 1b"
EMBEDDINGS_CACHE = "llama_embeddings_cache.npz"
PDF_PATHS = {
    0: "/Users/animeshsingh/Desktop/Datasets/Mental Health/depresion.pdf",
    1: "/Users/animeshsingh/Desktop/Datasets/Mental Health/anxity.pdf",
    2: "/Users/animeshsingh/Desktop/Datasets/Mental Health/bipolar.pdf",
    3: "/Users/animeshsingh/Desktop/Datasets/Mental Health/ocd.pdf",
    4: "/Users/animeshsingh/Desktop/Datasets/Mental Health/schiz.pdf",
}
LABEL_NAMES = ["depression", "anxiety", "bipolar", "ocd", "schizophrenia"]


class MCDropoutRouter(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_experts=NUM_EXPERTS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(self, x):
        return self.net(x)


def extract_pdf_text(path):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "".join(page.extract_text() or "" for page in reader.pages)


def to_examples(text, min_len=20):
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) < 2:
        parts = [s.strip() for s in text.split(". ") if s.strip()]
    return [p for p in parts if len(p) >= min_len]


def load_data():
    texts, labels = [], []
    rng = np.random.RandomState(42)
    for label, path in PDF_PATHS.items():
        examples = to_examples(extract_pdf_text(path))
        if len(examples) > MAX_PER_CLASS:
            idx = rng.choice(len(examples), MAX_PER_CLASS, replace=False)
            examples = [examples[i] for i in idx]
        texts.extend(examples)
        labels.extend([label] * len(examples))
    return texts, labels


def extract_embeddings(texts, model, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        inp = tokenizer(texts[i:i + EMBED_BATCH_SIZE], truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp, output_hidden_states=True).hidden_states[-1].float()
        mask = inp["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out, axis=0)


def get_embeddings(texts):
    if os.path.exists(EMBEDDINGS_CACHE):
        return np.load(EMBEDDINGS_CACHE)["X"]
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    X = extract_embeddings(texts, model, tokenizer)
    np.savez(EMBEDDINGS_CACHE, X=X)
    return X


def train():
    texts, labels = load_data()
    y = np.array(labels)
    X = get_embeddings(texts)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42,
                                              stratify=y)
    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                          batch_size=BATCH_SIZE, shuffle=True)

    counts = np.bincount(y_tr)
    weights = torch.FloatTensor(len(y_tr) / (NUM_EXPERTS * counts)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    router = MCDropoutRouter().to(DEVICE)
    opt = torch.optim.Adam(router.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        router.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            criterion(router(xb), yb).backward()
            opt.step()

    router.eval()
    with torch.no_grad():
        preds = router(torch.FloatTensor(X_te).to(DEVICE)).argmax(-1).cpu().numpy()
    print(classification_report(y_te, preds, target_names=LABEL_NAMES))

    torch.save(router.state_dict(), "mc_router.pt")
    print("saved mc_router.pt")


if __name__ == "__main__":
    train()
