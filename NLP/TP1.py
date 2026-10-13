import re
import string

import nltk
import pandas as pd
import spacy
import streamlit as st
from nltk.stem import SnowballStemmer
from nltk.tokenize import (
    TreebankWordTokenizer,
    TweetTokenizer,
    WordPunctTokenizer,
    sent_tokenize,
    word_tokenize,
)
from spacy.tokens import Doc
from textblob import TextBlob

try:
    import contractions
except ImportError:  # optional dependency
    contractions = None

try:
    import emoji
except ImportError:  # optional dependency
    emoji = None


def load_resources():
   
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(pkg, quiet=True)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    stop_words = set(nltk.corpus.stopwords.words("english"))
    return nlp, stop_words


NLP, STOP_WORDS = load_resources()
STEMMER = SnowballStemmer("english")


SENTENCE_TOKENIZERS = ["split", "NLTK", "spaCy", "TextBlob"]

WORD_TOKENIZERS = [
    "split()",
    "NLTK word_tokenize",
    "NLTK WordPunctTokenizer",
    "NLTK TreebankWordTokenizer",
    "NLTK TweetTokenizer",
    "spaCy",
    "TextBlob",
]

NORMALIZATIONS = [
    "replace emojis",
    "fix contractions",
    "spelling",
    "lowercasing",
    "remove punctuation",
    "remove extra spaces",
]


def word_tokenizer(text, tokenizer_type):
    
    if tokenizer_type == "split()":
        tokens = text.split()
    elif tokenizer_type == "NLTK word_tokenize":
        tokens = word_tokenize(text)
    elif tokenizer_type == "NLTK WordPunctTokenizer":
        tokens = WordPunctTokenizer().tokenize(text)
    elif tokenizer_type == "NLTK TreebankWordTokenizer":
        tokens = TreebankWordTokenizer().tokenize(text)
    elif tokenizer_type == "NLTK TweetTokenizer":
        tokens = TweetTokenizer().tokenize(text)
    elif tokenizer_type == "spaCy":
        tokens = [t.text for t in NLP.make_doc(text)]
    elif tokenizer_type == "TextBlob":
        tokens = list(TextBlob(text).words)
    else:
        raise ValueError(f"Unknown word tokenizer: {tokenizer_type}")
    return [t for t in tokens if t.strip()]


def sentence_tokenizer(text, tokenizer_type):
    """Split `text` into sentences using the chosen method."""
    if tokenizer_type == "split":
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    elif tokenizer_type == "NLTK":
        sentences = sent_tokenize(text)
    elif tokenizer_type == "spaCy":
        sentences = [s.text for s in NLP(text).sents]
    elif tokenizer_type == "TextBlob":
        sentences = [str(s) for s in TextBlob(text).sentences]
    else:
        raise ValueError(f"Unknown sentence tokenizer: {tokenizer_type}")
    return [s.strip() for s in sentences if s.strip()]


def normalize(text, operation):
    """Apply one cleaning / transformation operation to `text`."""
    if operation == "lowercasing":
        return text.lower()

    if operation == "fix contractions":
        if contractions is not None:
            return contractions.fix(text)
     
        fallback = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "cannot",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'ve\b": " have",
            r"'ll\b": " will",
            r"'d\b": " would",
            r"'m\b": " am",
            r"\blet's\b": "let us",
        }
        for pattern, repl in fallback.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    if operation == "remove punctuation":
        return text.translate(str.maketrans("", "", string.punctuation))

    if operation == "remove extra spaces":
        return re.sub(r"\s+", " ", text).strip()

    if operation == "replace emojis":
        if emoji is None:
            return text
        converted = emoji.demojize(text, delimiters=(" ", " "))
        return converted.replace("_", " ")

    if operation == "spelling":
        return str(TextBlob(text).correct())

    raise ValueError(f"Unknown normalization: {operation}")


def remove_stop_words(tokens):
   
    return [t for t in tokens if t.lower() not in STOP_WORDS]


def _spacy_doc_from_tokens(tokens):
    
    doc = Doc(NLP.vocab, words=tokens)
    for _, component in NLP.pipeline:
        doc = component(doc)
    return doc


def POS_tag(tokens):
  
    if not tokens:
        return []
    return [t.pos_ for t in _spacy_doc_from_tokens(tokens)]


def lemmatize(tokens):
   
    if not tokens:
        return []
    return [t.lemma_ for t in _spacy_doc_from_tokens(tokens)]


def stem(tokens):
  
    return [STEMMER.stem(t) for t in tokens]


st.title("NLP : Pre-processing")

text = st.text_area(
    "Entrez un texte :",
    value="Natural Language Processing is amazing. Let's go!",
)

sent_type = st.selectbox("Choisir le sentence tokenizer :", SENTENCE_TOKENIZERS, index=1)
word_type = st.selectbox("Choisir le word tokenizer :", WORD_TOKENIZERS, index=1)

normalizations = st.multiselect(
    "Appliquer la normalisation :",
    NORMALIZATIONS,
    default=["lowercasing", "fix contractions", "remove punctuation", "remove extra spaces"],
)

remove_sw = st.checkbox("Supprimer les stopwords")
use_pos = st.checkbox("POS Tagging")
use_stem = st.checkbox("Stemming")
use_lemma = st.checkbox("Lemmatization")

run = st.button("Lancer le traitement")


if run:
    if not text.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        # a. Sentence segmentation
        sentences = sentence_tokenizer(text, sent_type)


        for i, sentence in enumerate(sentences, start=1):
            st.markdown("---")
            st.markdown(f"**Sentences :** `{i}`")  

            
            normalized = sentence
            for op in [o for o in NORMALIZATIONS if o in normalizations]:
                normalized = normalize(normalized, op)
            st.markdown(f"**Texte normalisé :** {normalized}")

         
            tokens = word_tokenizer(normalized, word_type)

 
            if remove_sw:
                tokens = remove_stop_words(tokens)

            if not tokens:
                st.info("Aucun token après le traitement.")
                continue

      
            df = pd.DataFrame({"Tokens": tokens})

         
            if use_pos:
                df["POS"] = POS_tag(tokens)

            if use_stem:
                df["Stems"] = stem(tokens)

            if use_lemma:
                df["Lemms"] = lemmatize(tokens)

            st.dataframe(df, use_container_width=True)