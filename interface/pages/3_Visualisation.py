import streamlit as st
import streamlit.components.v1 as components
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

import tempfile
import os

import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster import hierarchy as sch
from scipy.cluster.hierarchy import fcluster

from sentence_transformers import SentenceTransformer

from typing import Dict, Tuple, List

# ✅ Élargir la largeur de l'application
st.markdown(
    """
    <style>
        .st-emotion-cache-13ln4jf {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Récupération des données du session_state
relations_corpus = st.session_state["relations_corpus"]

# Fonction pour construire un graphe pondéré
@st.cache_data
def build_weighted_graph(relations_dict: Dict[str, List[Tuple[str, str, str]]], min_weight=2):
    G = nx.Graph()
    edge_weights = defaultdict(int)

    for relations in relations_dict.values():
        for subj, rel, enf in relations:
            edge = (subj, enf)
            edge_weights[edge] += 1  # Compter les occurrences de chaque relation

    for (subj, enf), weight in edge_weights.items():
        if weight >= min_weight:
            G.add_edge(subj, enf, weight=weight)

    return G

# Fonction pour afficher le graphe avec Pyvis
def plot_interactive_graph(G):
    net = Network(notebook=False, height="500px", width="100%")
    
    # Ajouter les nœuds et les arêtes au graphe Pyvis
    for node in G.nodes():
        net.add_node(node, label=node, color="lightblue")

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v, value=weight, title=f"Poids: {weight}")

    # Générer un fichier HTML temporaire pour afficher le graphe
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_filename = tmp_file.name

    # Charger le graphe HTML dans Streamlit
    with open(tmp_filename, "r", encoding="utf-8") as f:
        html_graph = f.read()
    
    components.html(html_graph, height=550, scrolling=True)

    # Supprimer le fichier temporaire après utilisation
    os.remove(tmp_filename)

# Construire et afficher le graphe pondéré
G = build_weighted_graph(relations_corpus, min_weight=1)
st.title("🧑‍🔬 Visualisation du Graphe des Relations")
plot_interactive_graph(G)

# Calcul de la matrice de similarité des concepts
concepts = st.session_state.get("concepts_affiches", [])
embedding_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
embedding_concepts = embedding_model.encode(concepts, normalize_embeddings=True)
similarity_matrix = cosine_similarity(embedding_concepts)

# Création de la heatmap de similarité
st.subheader("🌡️ Heatmap de Similarité Sémantique")
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu",
    xticklabels=concepts,
    yticklabels=concepts,
    vmin=0,
    vmax=1
)
plt.title("Similarité sémantique entre les concepts", pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
st.pyplot(plt)

# Dendrogramme de clustering hiérarchique
st.subheader("📊 Dendrogramme des Concepts Sémantiques")
distance_matrix = 1 - similarity_matrix
linked = sch.linkage(distance_matrix, method='ward')

plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(
    linked,
    labels=concepts,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True
)
plt.title("Dendrogramme des concepts sémantiques", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel("Distance", fontsize=12)
plt.tight_layout()
st.pyplot(plt)

st.subheader("🔍 Synonymes Détectés")
similarity_threshold = st.slider(
    "Sélectionner le seuil de similarité pour les synonymes", 
    0.0, 1.0, 0.3, 0.01
)

# Conversion des distances en clusters
synonym_clusters = fcluster(linked, t=similarity_threshold, criterion='distance')

synonyms_dict = defaultdict(list)
for concept, cluster_id in zip(concepts, synonym_clusters):
    synonyms_dict[cluster_id].append(concept)
for cluster, synos in synonyms_dict.items():
    if len(synos) > 1:
        st.write(f"Cluster {cluster}: {', '.join(synos)}")


# Graphe orienté pour représenter la relation de méronymie
st.subheader("⚙️ Relations Méronymiques")
# Slider pour ajuster le seuil de méronymie
meronymy_threshold = st.slider(
    "Sélectionner le seuil de méronymie", 
    0.0, 1.0, 0.7, 0.01
)
G_meronymy = nx.DiGraph()

for i, (concept1, concept2) in enumerate(zip(concepts, concepts[1:])):
    distance = distance_matrix[i, i + 1]
    if distance < meronymy_threshold:
        G_meronymy.add_edge(concept1, concept2)

st.write("Relations méronymiques détectées :")
for edge in G_meronymy.edges:
    st.write(f"• {edge[0]} → {edge[1]}")

if G_meronymy:
    plot_interactive_graph(G_meronymy)