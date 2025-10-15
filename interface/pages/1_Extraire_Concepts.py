# ============ Import ============
from typing import List, Dict

import os
import streamlit as st
from pathlib import Path

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

import math

from collections import Counter

import nltk
import spacy
from nltk.util import ngrams
nltk.download('stopwords')
nltk.download('punkt')

# ============ Objets Global ============
global puncts
puncts = re.sub(r"[-']", r"", string.punctuation + "“”«»•")
global nlp
# Charger le modèle Spacy
nlp = spacy.load("fr_core_news_sm")
st.session_state["nlp"] = nlp

# Définir une expression régulière pour détecter les mots composés avec un tiret
hyphenated_word_pattern = re.compile(r"^[\w]+(?:-[\w]+)+$")

# Ajouter une règle de tokenisation pour conserver les mots composés avec des tirets
nlp.tokenizer.token_match = hyphenated_word_pattern.match

# ============ Fonctions ============

@st.cache_data
def data_loader(data_path:str)->Dict:
    name_corpus_brut = {}
    # Parcourir tous les fichiers .txt dans le répertoire spécifié
    for text in Path(data_path).rglob("*.txt"):
        with open(text, 'r', encoding='utf-8') as f:
            # Ajouter le texte à la liste après nettoyage des apostrophes et suppression des espaces superflus
            name_corpus_brut[text.stem] = f.read().replace("’", "'").strip()
    return name_corpus_brut

@st.cache_data
def data_preprocessing(corpus_brut:List[str], stopwords=False, majuscules=True)->List[List[str]]:
    
    def text_preprocessing(text:str)->List[str]:
        # Retirer la ponctuation spécifiée du texte
        text = re.sub(rf"[{puncts}]", "", text)
        # Nettoyer les espaces multiples et les ponctuations supplémentaires
        text = re.sub(r'\s+', ' ', re.sub(rf"[{puncts}]", "", text)).strip()
        # Appliquer le modèle Spacy pour obtenir les tokens du texte
        doc = nlp(text)
        if stopwords:
            # Lemmatiser les mots et enlever les espaces et les caractères indésirables si stopwords est True
            tokens = [token.lemma_.strip() for token in doc if not token.is_space and token.text not in "-'"]
        else:
            # Lemmatiser, enlever les stopwords, les espaces et les caractères indésirables si stopwords est False
            tokens = [token.lemma_.strip() for token in doc if not token.is_stop and not token.is_space and token.text not in "-'"]
        if not majuscules:
            # Mettre tous les tokens en minuscules si la variable majuscules est False
            tokens = [token.lower() for token in tokens]
        return " ".join(tokens)

    # Appliquer le prétraitement à chaque texte du corpus
    return [text_preprocessing(text) for text in corpus_brut]

def spacy_tokenizer(text):
    return [token.lemma_ for token in nlp(text)]

@st.cache_data
def keywords_TFIDF(n_grams: int = 2, seuil: float = 0.5, corpus_net = None) -> List[str]:
    """
    Extrait les mots-clés en utilisant TF-IDF sur un texte nettoyé.
    """

    # Si aucun corpus n'est passé, renvoyer une liste vide
    if not corpus_net:
        return []

    # Créer un vecteur TF-IDF en spécifiant la plage des n-grammes et le tokenizer Spacy
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(n_grams, n_grams), tokenizer=spacy_tokenizer)
    
    # Si le corpus est une liste, appliquer le fit_transform pour créer la matrice TF-IDF
    if isinstance(corpus_net, list):
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_net)
    else:
        # Si le corpus est un seul texte, l'encapsuler dans une liste
        tfidf_matrix = tfidf_vectorizer.fit_transform([corpus_net])

    # Obtenir les mots-clés en n-grams générés par le vecteur TF-IDF
    keywords = tfidf_vectorizer.get_feature_names_out()
    
    # Calculer la somme des scores TF-IDF pour chaque mot-clé dans le corpus
    scores = tfidf_matrix.sum(axis=0).A1

    # Trier les mots-clés en fonction de leurs scores, du plus élevé au plus bas
    sorted_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)

    return [pair[0] for pair in sorted_keywords if pair[1] >= seuil]

@st.cache_data
def keywordsPMI(corpus, n=2):
    # Tokeniser le corpus
    total_words = len(corpus)
    
    # Générer les n-grams
    ngram_list = list(ngrams(corpus, n))
    
    # Calculer les fréquences relatives
    word_freq = Counter(corpus)
    ngram_freq = Counter(ngram_list)
    
    # Calculer les probabilités de tokens
    p_word = {word: freq / total_words for word, freq in word_freq.items()}
    
    # Calculer les probabilité de n-grams
    total_ngrams = sum(ngram_freq.values())
    p_ngram = {ngram: freq / total_ngrams for ngram, freq in ngram_freq.items()}
    
    # Calculer le PMI
    pmi_scores = {}
    for ngram in ngram_freq:
        prob_ngram = p_ngram[ngram]
        prob_word_product = 1

        for word in ngram:  
            prob_word_product *= p_word.get(word, 1e-10)
        
        if prob_word_product > 0:
            pmi_scores[" ".join(ngram)] = math.log2(prob_ngram / prob_word_product)
        else:
            pmi_scores[" ".join(ngram)] = 0

    sorted_pmi_scores = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [pair[0] for pair in sorted_pmi_scores]

    return keywords

@st.cache_data
def keywordsCValue(corpus, n=2):
    
    ngram_list = list(ngrams(corpus, n))
    
    ngram_freq = Counter(ngram_list)
    
    ngram_lengths = {ngram: len(ngram) for ngram in ngram_freq}
    
    ngrams_containing = {ngram: [] for ngram in ngram_freq}
    
    # Créer un dictionnaire d'index des n-grammes
    ngram_set = set(ngram_freq.keys())
    
    for ngram in ngram_freq:
        # Chercher seulement dans les n-grammes plus grands
        for larger_ngram in ngram_set:
            if ngram != larger_ngram and set(ngram).issubset(set(larger_ngram)):
                ngrams_containing[ngram].append(larger_ngram)
    
    ngram_to_C_value = {}
    
    for ngram in ngram_freq:
        
        ngrams_containing_this_ngram = ngrams_containing[ngram]
        num_containing = len(ngrams_containing_this_ngram)
        
        freq_ngram = ngram_freq[ngram]
        length_ngram = ngram_lengths[ngram]
        
        if num_containing > 0:
            # Calcul de la somme des fréquences des n-grammes qui contiennent le n-gramme actuel
            sum_freq = sum(ngram_freq[ngram_cont] for ngram_cont in ngrams_containing_this_ngram)
            
            # Calcul du C-Value
            C_value = (math.log2(length_ngram) * freq_ngram) - (sum_freq / num_containing)
        else:
            # Si le n-gramme n'est contenu dans aucun autre n-gramme, calculer seulement la valeur de base
            C_value = math.log2(length_ngram) * freq_ngram
        
        ngram_to_C_value[ngram] = C_value
    
    sorted_ngram_C_values = sorted(ngram_to_C_value.items(), key=lambda x: x[1], reverse=True)
    
    keywords = [" ".join(ngram) for ngram, _ in sorted_ngram_C_values]
    
    return keywords

@st.cache_data
def keywordsRake(max_n_grams, text, seuil=0.0) -> List[str]:
    
    if max_n_grams:
        r = Rake(language="french", max_length=max_n_grams)
    else:
        r = Rake(language="french")

    # Appliquer RAKE pour extraire les phrases-clés du texte
    r.extract_keywords_from_text(text)
    
    # Récupérer les phrases-clés, triées de la plus haute à la plus basse en fonction de leur score
    keywords = r.get_ranked_phrases_with_scores()
    
    # Filtrer les résultats en fonction du seuil
    filtered_keywords = [phrase for score, phrase in keywords if score >= seuil]
    
    return filtered_keywords

@st.cache_data
def filtrer_concepts(concepts: List[str]):
    # Liste pour stocker les concepts filtrés
    filtered_concepts = []

    # Parcours de chaque concept dans la liste fournie
    for concept in concepts:
        # Traiter le concept avec le modèle spacy pour effectuer le traitement linguistique
        doc = nlp(concept)
        
        # Initialiser un flag pour déterminer si le concept doit être gardé
        flag = True
        
        # Vérifier chaque token du concept
        for token in doc:
            # Si le token est un verbe, un adverbe ou un stop word, le concept est rejeté
            if token.pos_ in ['VERB', 'ADV'] or token.is_stop or token.is_punct:
                flag = False
        
        # Si aucun mot du concept ne l'empêche d'être gardé, l'ajouter à la liste filtrée
        if flag:
            filtered_concepts.append(concept)
    
    return filtered_concepts

def find_concepts_in_text(concepts: List[str], text_brut: str) -> List[str]:
    # Initialiser la liste des concepts trouvés
    found_concepts = []
    
    # Parcours de chaque concept à rechercher dans le texte
    for concept in concepts:
        flag = True
        
        # Vérifier si chaque token du concept est présent dans le texte
        for token in concept.split():
            # Échapper les caractères spéciaux du token et créer une expression régulière pour le rechercher
            concept_pattern = re.escape(token)
            
            # Si un des tokens du concept n'est pas trouvé dans le texte, marquer le concept comme non trouvé
            if not re.search(rf'\b{concept_pattern}\b', text_brut):
                flag = False
        
        # Si tous les tokens du concept sont trouvés dans le texte, ajouter le concept à la liste des concepts trouvés
        if flag:
            found_concepts.append(concept)
    
    return found_concepts

def surligner_in_streamlit(concepts: List[str], text_brut: str) -> str:
    text = text_brut
    for concept in concepts:
        for token in concept.split():
            concept_pattern = re.escape(token)
            text = re.sub(rf"(\s|^)({concept_pattern})(\s|$)", r"\1<mark>\2</mark>\3", text, flags=re.IGNORECASE)
    return text
        
def main():
    st.title("Extraction de Concepts")

    # Créer un formulaire
    with st.form(key="folder_form"):
        folder_path_brut = st.text_input("Entrez le chemin du dossier :")
        # Bouton pour soumettre le formulaire
        submit_button = st.form_submit_button(label="Valider")
    folder_path = "../" + folder_path_brut # Assurer qu'on part toujours du "Home.py"

    # Le chemin n'est récupéré que si le bouton est cliqué
    if submit_button:
        st.write(f"Chemin entré : {folder_path_brut}")
        
    # Obtenir le chemin absolu du dossier courant (où se trouve script.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if folder_path_brut:
        # Construire le chemin absolu du dossier corpus
        corpus_path = os.path.abspath(os.path.join(current_dir, folder_path))
        st.session_state["corpus_path"] = corpus_path
    else:
        corpus_path = ""

    corpus_not_found_flag=False
    if "corpus_path" in st.session_state:
        corpus_path = st.session_state["corpus_path"]
        st.write(f"📂 Chemin complet du dossier corpus entré : `{corpus_path}`")
        # Vérifier si le dossier existe
        if os.path.isdir(corpus_path):
            pass
        else:
            st.error("Dossier introuvable ❌")
            corpus_not_found_flag=True
            corpus_path = ""
    
    if "name_corpus_brut" not in st.session_state:
        st.session_state["name_corpus_brut"] = ""
    if corpus_path:
        name_corpus_brut = data_loader(corpus_path)
        st.session_state["name_corpus_brut"] = name_corpus_brut
    else:
        name_corpus_brut = st.session_state["name_corpus_brut"]
    
    if name_corpus_brut and not corpus_not_found_flag:
        corpus_brut = list(name_corpus_brut.values())
        corpus_net = data_preprocessing(corpus_brut, stopwords=False)
        st.session_state["corpus_brut"] = corpus_brut
        st.session_state["corpus_net"] = corpus_net

        # Choix de la façon d'extraction
        method = st.selectbox(
                    'Choisissez la méthode d\'extraction',
                    ('', 'TF-IDF', 'PMI', 'C-Value', 'Rake'))
        if method:
            st.session_state["method"] = method
        if "method" in st.session_state:
            method = st.session_state["method"]
        if method:
            st.markdown(f"<strong>Méthode Choisie : {method}</strong>", unsafe_allow_html=True)
            if method == "TF-IDF":
                # Détection des concepts avec TF-IDF
                n_grams = st.slider("Taille des n-grammes", 1, 4, st.session_state['n_grams'] if 'n_grams' in st.session_state else 2, step=1)
                st.session_state['n_grams'] = n_grams
                seuil = st.slider("Seuil TF-IDF", 0.0, 1.0, st.session_state['seuil'] if 'seuil' in st.session_state else 0.1, step=0.1)
                st.session_state['seuil'] = seuil
                concepts_extraits = keywords_TFIDF(n_grams, seuil, corpus_net)
                concepts_extraits = filtrer_concepts(concepts_extraits)

            elif method == "PMI":
                # Détection des concepts avec PMI
                n_grams = st.slider("Taille des n-grammes", 1, 4, st.session_state['n_grams'] if 'n_grams' in st.session_state else 2, step=1)
                st.session_state['n_grams'] = n_grams
                nb_keywords = st.slider("Nombre de résultats", 10, 100, st.session_state['nb_keywords'] if 'nb_keywords' in st.session_state else 50, step=10)
                st.session_state['nb_keywords'] = nb_keywords
                concepts_extraits = keywordsPMI(" ".join(corpus_net).split(), n_grams)
                concepts_extraits = filtrer_concepts(concepts_extraits)[:nb_keywords]

            elif method == "C-Value":
                # Détection des concepts avec C-Value
                n_grams = st.slider("Taille des n-grammes", 1, 4, st.session_state['n_grams'] if 'n_grams' in st.session_state else 2, step=1)
                st.session_state['n_grams'] = n_grams
                nb_keywords = st.slider("Nombre de résultats (⚠️ ça prend environ 10 minutes)", 10, 100, st.session_state['nb_keywords'] if 'nb_keywords' in st.session_state else 50, step=10)
                st.session_state['nb_keywords'] = nb_keywords
                concepts_extraits = keywordsCValue(" ".join(corpus_net).split(), n_grams)
                concepts_extraits = filtrer_concepts(concepts_extraits)[:nb_keywords]

            else:
                # Détection des concepts avec Rake
                n_grams = st.slider("Taille de max n-grammes", 1, 4, 2)
                seuil_rake = st.slider("Seuil RAKE", 0.0, 10.0, st.session_state['seuil_rake'] if 'seuil_rake' in st.session_state else 0.1, step=1.0)
                st.session_state['seuil_rake'] = seuil_rake
                nb_keywords = st.slider("Nombre de résultats", 10, 100, st.session_state['nb_keywords'] if 'nb_keywords' in st.session_state else 50, step=10)
                st.session_state['nb_keywords'] = nb_keywords
                concepts_extraits = keywordsRake(n_grams, "\n".join(corpus_brut), seuil_rake)
                concepts_extraits = filtrer_concepts(concepts_extraits)[:nb_keywords]

            concepts_affiches = concepts_extraits
            if st.button("Raffiner les concepts", key="Raffiner les concepts") and ('refine_concepts' not in st.session_state or ('refine_concepts' in st.session_state and not st.session_state['refine_concepts'])):
                st.session_state['refine_concepts'] = True
                # Trouver les concepts dans le corpus brut
                filtered_concepts = find_concepts_in_text(concepts_extraits, "\n".join(corpus_brut))
                concepts_affiches = filtered_concepts
                st.session_state['filtered_concepts'] = filtered_concepts
            if st.button("Ne pas raffiner les concepts", key="Cancel : Raffiner les concepts") and ('refine_concepts' in st.session_state and st.session_state['refine_concepts']):
                concepts_affiches = concepts_extraits
                st.session_state['refine_concepts'] = False
                
            # Afficher les concepts détectés
            st.subheader("Concepts extraits")
            if 'refine_concepts' in st.session_state and st.session_state['refine_concepts']:
                concepts_affiches = st.session_state['filtered_concepts']
            st.write(concepts_affiches)
            st.session_state['concepts_affiches'] = concepts_affiches
            # Surligner les concepts dans le texte
            st.subheader("Texte avec concepts surlignés")
            corpus_name = st.selectbox(
                    'Choisissez un texte à charger',
                    list(name_corpus_brut.keys()))
            corpus_a_charger = name_corpus_brut[corpus_name]
            highlighted_text = surligner_in_streamlit(concepts_affiches, corpus_a_charger)
            st.markdown(highlighted_text, unsafe_allow_html=True)
            st.session_state["highlighted_text"] = highlighted_text

# ============ Exécution ============
if __name__ == "__main__":
    main()