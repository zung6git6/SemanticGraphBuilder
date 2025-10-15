# ============ Import ============

import streamlit as st

# ============ Exécution ============

# Titre de l'application
st.title("Application d'Extraction et de Visualisation d'Ontologies")

# Introduction
st.markdown("""
Bienvenue dans l'application dédiée à l'extraction de concepts et de relations à partir de votre corpus textuel. Cette application vous permet de :

1. **Extraire des concepts** :
   L'application analyse le texte fourni pour extraire des concepts clés avec possibilité de les raffiner, qui peuvent être utilisés pour construire une ontologie sémantique.

2. **Extraire des relations entre concepts** :
   Une fois les concepts extraits, l'application identifie et affiche les relations sémantiques entre ces concepts, telles que des relations de type « est-un », « partie-de », etc.

3. **Visualiser l'ontologie des relations** :
   Vous pouvez explorer les relations entre les concepts à travers des graphes interactifs. Il est également possible de visualiser une heatmap de similarité sémantique ainsi qu’un dendrogramme des concepts.
   En complément, les concepts synonymes et méronymes sont disponibles, avec la possibilité de modifier les seuils de critères pour affiner ces informations.

4. **Exporter l'ontologie** :
   L'application permet également d'exporter l'ontologie sous différents formats comme **RDF** ou **OWL**.

---

### Quick Start !

1. Indiquez le chemin vers votre corpus de textes.
2. L'application extraira les concepts et les relations à partir de votre texte.
3. Ensuite, vous pourrez visualiser l'ontologie sous forme de graphe interactif et analyser les concepts, les relations.
4. Finalement, Vous pourrez exporter l'ontologie au format RDF ou OWL selon vos besoins.

---

### Let's GO !

Utilisez le menu à gauche pour charger votre corpus et commencer l'extraction. Vous pouvez aussi explorer l'ontologie générée et l'exporter dans le format de votre choix.

""")