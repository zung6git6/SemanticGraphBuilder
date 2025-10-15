# ============ Import ============
import streamlit as st

from typing import List, Dict, Tuple

# ============ Exécution ============

# Chargement du modèle NLP et récupération des données du session state
nlp = st.session_state.get("nlp", None)
concepts = st.session_state.get("concepts_affiches", [])
name_corpus_brut = st.session_state.get("name_corpus_brut", "Corpus inconnu")
corpus_brut = st.session_state.get("corpus_brut", [])
corpus_net = st.session_state.get("corpus_net", [])

# Vérification de la présence du modèle NLP
if not nlp:
    st.error("⚠️ Modèle NLP non chargé dans `st.session_state`. Veuillez vérifier le chargement.")
    st.stop()

st.title("🔍 Extraction de Relations dans un Corpus")

st.sidebar.write(f"📄 **Nombre de documents** : `{len(corpus_brut)}`")

# Fonction pour trouver le sujet d'une phrase
@st.cache_data
def find_subject(sentence: str) -> int:
    doc = nlp(sentence)
    for token in doc:
        if "nsubj" in token.dep_:
            return token.i
    return -1  # Retourner -1 si aucun sujet trouvé

@st.cache_data
def extract_relations(sentence: str, concepts: List[str]) -> List[Tuple[str, str, str]]:
    concepts_tokenised = [word for c in concepts for word in c.split()]
    subject_idx = find_subject(sentence)
    
    if subject_idx == -1:
        return []

    doc = nlp(sentence)
    subject = doc[subject_idx].text
    subject_lemma = doc[subject_idx].lemma_
    
    # Vérifier si le sujet est un concept
    flag = subject_lemma in concepts_tokenised
    relations = []
    
    for token in doc:
        if token.head.i == subject_idx and (flag or token.lemma_ in concepts_tokenised):
            relations.append((subject, token.dep_, token.text))
    
    return relations

@st.cache_data
def process_corpus(concepts_extraits: List[str], corpus_brut: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
    relations_dict = {}
    
    progress_bar = st.progress(0)
    total_sentences = sum(len(list(nlp(doc).sents)) for doc in corpus_brut)
    processed_sentences = 0

    for corpus in corpus_brut:
        doc = nlp(corpus)
        for sentence in doc.sents:
            relations = extract_relations(sentence.text, concepts_extraits)
            if relations:
                relations_dict[sentence.text] = relations
            
            # Mettre à jour la barre de progression
            processed_sentences += 1
            progress_bar.progress(processed_sentences / total_sentences)

    progress_bar.empty()  # Supprimer la barre de progression après exécution
    return relations_dict

relations_corpus = process_corpus(concepts, corpus_brut)
st.session_state["relations_corpus"] = relations_corpus

st.subheader("📌 Relations extraites")

if relations_corpus:
    with st.expander("🔽 Afficher/Masquer les relations détectées"):
        for sentence, relations in relations_corpus.items():
            st.write(f"**🔹 Phrase :** `{sentence}`")
            st.table([{"Sujet": r[0], "Relation": r[1], "Objet": r[2]} for r in relations])
else:
    st.info("Aucune relation détectée dans le corpus.")

st.success("✅ Extraction terminée avec succès !")