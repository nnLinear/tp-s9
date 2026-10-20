
import re
import math
import numpy as np


try:
    import nltk
    for res in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(res, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(stopwords.words("english"))
    _lem = WordNetLemmatizer()
    _lem.lemmatize("tests") 

    def lemmatize(w):
       
        v = _lem.lemmatize(w, pos="v")
        return v if v != w else _lem.lemmatize(w, pos="n")
except Exception:
    STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to",
                  "in", "on", "at", "with", "for", "by", "it", "this", "that", "his",
                  "her", "he", "she", "they", "them", "very", "from", "as", "be"}

    def lemmatize(w):  # lemmatiseur minimal de secours
        for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                base = w[: -len(suf)] + rep
                if suf in ("ing", "ed") and len(base) > 2 and base[-1] == base[-2]:
                    base = base[:-1]
                return base
        return w


def preprocess(sentence):
    tokens = re.findall(r"[A-Za-z]+", sentence)         
    tokens = [t.lower() for t in tokens]                 
    tokens = [t for t in tokens if t not in STOP_WORDS]  
    return [lemmatize(t) for t in tokens]                


def build_vocab(docs):
    words = sorted({w for d in docs for w in d})
    return {w: i for i, w in enumerate(words)}


def one_hot(index, size):
    v = np.zeros(size)
    v[index] = 1
    return v


corpus = [
    "The king loves the queen and the queen loves the king",
    "The man walks with the woman in the garden",
    "The king and the man are strong, the queen and the woman are wise",
    "Children are playing in the garden with the dogs",
    "The dogs are running and the children are laughing",
]
docs = [preprocess(s) for s in corpus]
vocab = build_vocab(docs)
V = len(vocab)
inv_vocab = {i: w for w, i in vocab.items()}

print("Phrases prétraitées :")
for d in docs:
    print("  ", d)
print("Vocabulaire :", vocab, "\n")


print("=" * 70, "\nEXERCICE 1 : one-hot\n" + "=" * 70)
onehot_corpus = [[one_hot(vocab[w], V) for w in d] for d in docs]
print("One-hot de la première phrase :")
for w, vec in zip(docs[0], onehot_corpus[0]):
    print(f"  {w:10s} -> {vec.astype(int)}")

#ex2
print("\n" + "=" * 70, "\nEXERCICE 2 : Bag of Words\n" + "=" * 70)
bow = []
for d in docs:
    vec = np.zeros(V, dtype=int)
    for w in d:
        vec[vocab[w]] += 1
    bow.append(vec)
print("Colonnes :", list(vocab))
for i, vec in enumerate(bow):
    print(f"  Phrase {i + 1} : {vec}")

# ---------------------------------------------------------------------------
# Exercice 3 : TF-IDF
# ---------------------------------------------------------------------------
print("\n" + "=" * 70, "\nEXERCICE 3 : TF-IDF\n" + "=" * 70)
N = len(docs)
TF = np.array([vec / vec.sum() for vec in bow])           
df = (np.array(bow) > 0).sum(axis=0)                     
IDF = np.log(N / df)                                      
TFIDF = TF * IDF                                         

np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("Vocabulaire :", vocab)
print("\nMatrice TF :\n", TF)
print("\nVecteur IDF :\n", IDF)
print("\nMatrice TF-IDF :\n", TFIDF)


#skip gram
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


window_size = 2
embedding_size = 10
lr = 0.05
epochs = 2000
tokens = [w for d in docs for w in d]  


#ex3
print("\n" + "=" * 70, "\nEXERCICE 4 : CBOW\n" + "=" * 70)


cbow_data = []
for i, target in enumerate(tokens):
    ctx_idx = [j for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1))
               if j != i]
    ctx_vecs = [one_hot(vocab[tokens[j]], V) for j in ctx_idx]
    x = np.mean(ctx_vecs, axis=0)                
    y = one_hot(vocab[target], V)                 
    cbow_data.append((x, y))

# 4. Matrices
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.1, (V, embedding_size))
W2 = rng.normal(0, 0.1, (embedding_size, V))

# 5-6. Entraînement
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for x, y in cbow_data:
        h = x @ W1                                # forward
        u = h @ W2
        y_hat = softmax(u)                        # softmax
        total_loss += -np.sum(y * np.log(y_hat + 1e-9))   # entropie croisée
        e = y_hat - y                             # rétropropagation
        dW2 = np.outer(h, e)
        dW1 = np.outer(x, e @ W2.T)
        W1 -= lr * dW1                            # mise à jour
        W2 -= lr * dW2
    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d} | loss = {total_loss:.4f}")

emb_cbow = W1.copy()
print("\nEmbeddings CBOW (quelques mots) :")
for w in ["king", "queen", "man", "woman", "garden"]:
    if w in vocab:
        print(f"  {w:8s}", emb_cbow[vocab[w]])

#ex5
print("\n" + "=" * 70, "\nEXERCICE 5 : Skip-gram\n" + "=" * 70)

# 3. Paires 
inputs, outputs = [], []
for i, target in enumerate(tokens):
    t_vec = one_hot(vocab[target], V)
    ctx_vecs = [one_hot(vocab[tokens[j]], V)
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1))
                if j != i]
    inputs.append(t_vec)
    outputs.append(ctx_vecs)          # liste des contextes de cette cible

W1 = rng.normal(0, 0.1, (V, embedding_size))
W2 = rng.normal(0, 0.1, (embedding_size, V))

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for x, ctxs in zip(inputs, outputs):
        h = x @ W1
        y_hat = softmax(h @ W2)
        # somme des pertes sur tous les contextes de la cible
        total_loss += sum(-np.sum(c * np.log(y_hat + 1e-9)) for c in ctxs)
        # erreur cumulée : somme des (y_hat - c_j)
        e = sum(y_hat - c for c in ctxs)
        dW2 = np.outer(h, e)
        dW1 = np.outer(x, e @ W2.T)
        W1 -= lr * dW1
        W2 -= lr * dW2
    if epoch % 200 == 0:
        print(f"Epoch {epoch:5d} | loss = {total_loss:.4f}")

emb_sg = W1.copy()
print("\nEmbeddings Skip-gram (quelques mots) :")
for w in ["king", "queen", "man", "woman", "garden"]:
    if w in vocab:
        print(f"  {w:8s}", emb_sg[vocab[w]])


#ex5
print("\n" + "=" * 70, "\nEXERCICE 6 : similarité cosinus\n" + "=" * 70)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


pairs = [("king", "queen"), ("man", "woman"), ("king", "man"),
         ("queen", "woman"), ("king", "garden"), ("dog", "child")]
print(f"{'Paire':20s}{'CBOW':>10s}{'Skip-gram':>12s}")
for w1, w2 in pairs:
    if w1 in vocab and w2 in vocab:
        s_c = cosine_similarity(emb_cbow[vocab[w1]], emb_cbow[vocab[w2]])
        s_s = cosine_similarity(emb_sg[vocab[w1]], emb_sg[vocab[w2]])
        print(f"{w1 + ' / ' + w2:20s}{s_c:10.3f}{s_s:12.3f}")

