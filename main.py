import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# ——— 1) Helpers & Caching ——————————————————————————————————

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_bert(model_name: str = 'bert-base-uncased', num_labels: int = 6):
    tok = BertTokenizer.from_pretrained(model_name)
    mod = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type='single_label_classification'
    )
    mod.eval()
    return tok, mod

class TextVectorizer:
    def __init__(self, texts, max_vocab_size=20000, seq_len=200):
        self.seq_len = seq_len
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        most_common = counter.most_common(max_vocab_size)
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (tok, _) in enumerate(most_common, start=2):
            self.token2idx[tok] = i
        self.vocab_size = len(self.token2idx)

    def vectorize(self, text: str) -> torch.LongTensor:
        toks = text.split()
        idxs = [self.token2idx.get(w, 1) for w in toks]
        if len(idxs) >= self.seq_len:
            idxs = idxs[: self.seq_len]
        else:
            idxs += [0] * (self.seq_len - len(idxs))
        return torch.tensor(idxs, dtype=torch.long)

    def __call__(self, texts):
        return torch.stack([self.vectorize(t) for t in texts])

@st.cache_resource
def build_vectorizer(texts, max_vocab_size=20000, seq_len=200):
    return TextVectorizer(texts, max_vocab_size, seq_len)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0,2,1)
        c = torch.relu(self.conv(emb))
        p = self.pool(c).squeeze(2)
        return self.fc(p)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n.squeeze(0))

@st.cache_resource
def train_pytorch_model(model_class, _vectorizer, X_train, y_train, X_val, y_val,
                        num_classes, epochs=5, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_class is LSTMModel:
        model = model_class(_vectorizer.vocab_size, 128, 64, num_classes)
    else:
        model = model_class(_vectorizer.vocab_size, 128, num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = _vectorizer(X_train)
    X_val_t   = _vectorizer(X_val)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
                              batch_size=batch_size)

    history = {'train_acc':[], 'val_acc':[]}
    for _ in range(epochs):
        # train
        model.train()
        preds, labels = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            batch_preds = logits.argmax(1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(yb.cpu().numpy())
        history['train_acc'].append(accuracy_score(labels, preds))

        # validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                batch_preds = logits.argmax(1).cpu().numpy()
                preds.extend(batch_preds)
                labels.extend(yb.cpu().numpy())
        history['val_acc'].append(accuracy_score(labels, preds))

    return model, history

def predict_bert(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()

def predict_pytorch(texts, model, vectorizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    X = vectorizer(texts).to(device)
    with torch.no_grad():
        logits = model(X)
    return torch.softmax(logits, dim=1).cpu().numpy()

def plot_wordcloud(texts):
    wc = WordCloud(width=400, height=250, background_color="white")\
         .generate(" ".join(texts))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def plot_pytorch_history(history, title):
    fig, ax = plt.subplots()
    ax.plot(history['train_acc'], label='train_acc')
    ax.plot(history['val_acc'],   label='val_acc')
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); st.pyplot(fig)

# ——— 2) App Layout ————————————————————————————————————————————

st.title("🎵 Lyrics Emotion Analyzer + Detailed Metrics")

# Load & prep
data_path = st.sidebar.text_input("CSV path", "SingleLabel.csv")
df = load_data(data_path)
st.sidebar.write(f"Loaded {df.shape[0]} rows")
labels = sorted(df['label'].unique())
num_classes = len(labels)

label2idx = {l:i for i,l in enumerate(labels)}
y = df['label'].map(label2idx).values
X = df['lyrics'].values
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# placeholders
vectorizer = None
lstm_model = None
cnn_model  = None
lstm_hist  = None
cnn_hist   = None

# ——— 2.1 Sidebar: Train & Evaluate —————————————————————————————

if st.sidebar.button("▶️ Train & Evaluate CNN + LSTM"):
    st.sidebar.info("Training models… this may take a few minutes")
    vectorizer = build_vectorizer(df['lyrics'].values)

    # LSTM
    with st.spinner("Training LSTM…"):
        lstm_model, lstm_hist = train_pytorch_model(
            LSTMModel, vectorizer, X_train, y_train, X_val, y_val, num_classes
        )
    # CNN
    with st.spinner("Training CNN…"):
        cnn_model, cnn_hist = train_pytorch_model(
            CNNModel, vectorizer, X_train, y_train, X_val, y_val, num_classes
        )
    st.sidebar.success("✅ Training complete!")

    # ——— Evaluate BERT —————————————————————————————————————
    bert_tok, bert_mod = load_bert(num_labels=num_classes)
    bert_probs = predict_bert(X_val.tolist(), bert_tok, bert_mod)
    bert_preds = bert_probs.argmax(1)
    bert_acc   = accuracy_score(y_val,  bert_preds)
    bert_rep   = classification_report(
        y_val, bert_preds, target_names=labels, output_dict=True, zero_division=0
    )
    bert_cm    = confusion_matrix(y_val, bert_preds)
    bert_df    = pd.DataFrame(bert_rep).T

    # ——— Evaluate LSTM —————————————————————————————————————
    lstm_probs = predict_pytorch(X_val.tolist(), lstm_model, vectorizer)
    lstm_preds = lstm_probs.argmax(1)
    lstm_acc   = accuracy_score(y_val, lstm_preds)
    lstm_rep   = classification_report(
        y_val, lstm_preds, target_names=labels, output_dict=True, zero_division=0
    )
    lstm_cm    = confusion_matrix(y_val, lstm_preds)
    lstm_df    = pd.DataFrame(lstm_rep).T

    # ——— Evaluate CNN —————————————————————————————————————
    cnn_probs  = predict_pytorch(X_val.tolist(), cnn_model, vectorizer)
    cnn_preds  = cnn_probs.argmax(1)
    cnn_acc    = accuracy_score(y_val, cnn_preds)
    cnn_rep    = classification_report(
        y_val, cnn_preds, target_names=labels, output_dict=True, zero_division=0
    )
    cnn_cm     = confusion_matrix(y_val, cnn_preds)
    cnn_df     = pd.DataFrame(cnn_rep).T

    # ——— Summary Accuracy —————————————————————————————————————
    summary = pd.DataFrame({
        'Model': ['BERT','LSTM','CNN'],
        'Val Accuracy': [bert_acc, lstm_acc, cnn_acc]
    }).set_index('Model')
    st.subheader("📊 Validation Accuracy Comparison")
    st.bar_chart(summary)

    # ——— Per‑class Metrics Tables ——————————————————————————————
    st.subheader("📋 Per‑Class Metrics (Precision / Recall / F1‑Score / Support)")
    st.markdown("**BERT**")
    st.dataframe(bert_df.style.format({
        'precision':'{:.2f}', 'recall':'{:.2f}', 'f1-score':'{:.2f}', 'support':'{:.0f}'
    }))
    st.markdown("**LSTM**")
    st.dataframe(lstm_df.style.format({
        'precision':'{:.2f}', 'recall':'{:.2f}', 'f1-score':'{:.2f}', 'support':'{:.0f}'
    }))
    st.markdown("**CNN**")
    st.dataframe(cnn_df.style.format({
        'precision':'{:.2f}', 'recall':'{:.2f}', 'f1-score':'{:.2f}', 'support':'{:.0f}'
    }))

    # ——— Confusion Matrices ————————————————————————————————————
    def show_cm(cm, title):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        st.markdown(f"**{title}**")
        st.dataframe(df_cm)
    st.subheader("� Confusion Matrices")
    show_cm(bert_cm, "BERT Confusion Matrix")
    show_cm(lstm_cm, "LSTM Confusion Matrix")
    show_cm(cnn_cm,  "CNN Confusion Matrix")

# ——— 2.2 Inference Model Selector —————————————————————————————

model_choice = st.sidebar.selectbox("Inference Model", ["BERT","LSTM","CNN"])

# auto‑train if needed
if model_choice != "BERT" and vectorizer is None:
    vectorizer = build_vectorizer(df['lyrics'].values)
    if model_choice=="LSTM":
        lstm_model, _ = train_pytorch_model(
            LSTMModel, vectorizer, X_train, y_train, X_val, y_val, num_classes
        )
    else:
        cnn_model, _ = train_pytorch_model(
            CNNModel, vectorizer, X_train, y_train, X_val, y_val, num_classes
        )

# ——— 3) Snippet Prediction ——————————————————————————————————

st.header("🎤 Predict Emotion for a Lyric Snippet")
snippet = st.text_area("Enter some lyrics here:")
if st.button("Analyze"):
    if model_choice=="BERT":
        tok, mdl = load_bert(num_labels=num_classes)
        probs = predict_bert([snippet], tok, mdl)[0]
    else:
        mdl = lstm_model if model_choice=="LSTM" else cnn_model
        probs = predict_pytorch([snippet], mdl, vectorizer)[0]
    st.bar_chart(pd.DataFrame({'Emotion':labels,'Prob':probs}).set_index('Emotion'))

# ——— 4) Emotion Arc ————————————————————————————————————————

st.header("📈 Emotion Arc of a Full Song")
song = st.selectbox("Pick a song", df['title'].unique())
full = df.loc[df['title']==song,'lyrics'].iat[0]
lines = [l.strip() for l in full.split('\n') if l.strip()]
if lines:
    if model_choice=="BERT":
        tok, mdl = load_bert(num_labels=num_classes)
        scores = predict_bert(lines, tok, mdl)
    else:
        mdl = lstm_model if model_choice=="LSTM" else cnn_model
        scores = predict_pytorch(lines, mdl, vectorizer)
    arc_df = pd.DataFrame(scores, columns=labels, index=range(1,len(lines)+1))
    arc_df.index.name = "Line #"
    st.line_chart(arc_df)
else:
    st.warning("No lyric lines found.")

# ——— 5) Word Clouds ————————————————————————————————————————

st.header("☁️ Word Clouds by Emotion")
for emo in labels:
    st.subheader(emo)
    plot_wordcloud(df[df['label']==emo]['lyrics'])
