# ============ Import ============

import streamlit as st

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, OWL, RDFS

import os
import tempfile

import re

# ============ Exécution ============

# Récupération des relations depuis la session Streamlit
relations_corpus = st.session_state.get("relations_corpus", {})

st.title("Exportation d'Ontologie en Format RDF/OWL")

def clean_for_uri(text):
    # Remplacer les caractères non valides par "_", sauf si c'est une ponctuation à conserver
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text.strip())

def serialize_to_rdf(relations_corpus, output_file):
    g = Graph()
    EX = Namespace("http://example.org/")
    g.bind("ex", EX)
    
    for sentence, triplets in relations_corpus.items():
        for triplet in triplets:
            subject, relation, obj = triplet
            subj_uri = URIRef(EX[subject.replace(" ", "_")])
            pred_uri = URIRef(EX[relation.replace(" ", "_")])
            # Si l'objet est une ponctuation ou une phrase, le traiter comme littéral
            if re.match(r'^[.,!?;:()\[\]{}\'"]+$', obj.strip()) or " " in obj:
                obj_node = Literal(obj)
            else:
                obj_node = URIRef(EX[clean_for_uri(obj)])
            
            g.add((subj_uri, pred_uri, obj_node))
            g.add((subj_uri, EX["mentionedIn"], Literal(sentence)))
    
    g.serialize(destination=output_file, format="xml")

def serialize_to_owl(relations_corpus, output_file):
    g = Graph()
    EX = Namespace("http://example.org/")
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf", RDF)
    
    # Déclarer l'ontologie
    ontology_uri = URIRef(EX["MyOntology"])
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    
    # Déclarer les classes de base
    entity_class = URIRef(EX["Entite"])
    subject_class = URIRef(EX["Sujet"])
    object_class = URIRef(EX["Objet"])
    
    g.add((entity_class, RDF.type, OWL.Class))
    g.add((subject_class, RDF.type, OWL.Class))
    g.add((object_class, RDF.type, OWL.Class))
    
    # Définir les sous-classes
    g.add((subject_class, RDFS.subClassOf, entity_class))
    g.add((object_class, RDFS.subClassOf, entity_class))
    
    declared_properties = set()
    declared_individuals = set()
    
    for sentence, triplets in relations_corpus.items():
        for subject, relation, obj in triplets:
            subj_uri = URIRef(EX[clean_for_uri(subject)])
            pred_uri = URIRef(EX[clean_for_uri(relation)])
            
            if re.match(r'^[.,!?;:()\[\]{}\'"]+$', obj.strip()) or " " in obj:
                obj_node = Literal(obj)
            else:
                obj_node = URIRef(EX[clean_for_uri(obj)])
            
            # Déclarer le sujet comme individu et l'associer à la classe Sujet
            if subj_uri not in declared_individuals:
                g.add((subj_uri, RDF.type, OWL.NamedIndividual))
                g.add((subj_uri, RDF.type, subject_class))  # Associer à la classe Sujet
                declared_individuals.add(subj_uri)
            
            # Déclarer l'objet comme individu et l'associer à la classe Objet, si ce n'est pas un littéral
            if isinstance(obj_node, URIRef) and obj_node not in declared_individuals:
                g.add((obj_node, RDF.type, OWL.NamedIndividual))
                g.add((obj_node, RDF.type, object_class))  # Associer à la classe Objet
                declared_individuals.add(obj_node)
            
            # Déclarer la propriété
            if pred_uri not in declared_properties:
                if isinstance(obj_node, Literal):
                    g.add((pred_uri, RDF.type, OWL.DatatypeProperty))
                else:
                    g.add((pred_uri, RDF.type, OWL.ObjectProperty))
                declared_properties.add(pred_uri)
            
            g.add((subj_uri, pred_uri, obj_node))
            
            mentioned_in_uri = EX["mentionedIn"]
            if mentioned_in_uri not in declared_properties:
                g.add((mentioned_in_uri, RDF.type, OWL.AnnotationProperty))
                declared_properties.add(mentioned_in_uri)
            g.add((subj_uri, mentioned_in_uri, Literal(sentence)))
    
    g.serialize(destination=output_file, format="xml")


if not relations_corpus:
    st.warning("Aucune donnée à exporter.")
else:
    # Option de format de fichier (RDF ou OWL)
    format_option = st.selectbox("Choisissez le format d'exportation", ["RDF", "OWL"])

    # Bouton pour exporter l'ontologie
    if st.button("Exporter l'ontologie"):
        with st.spinner("Génération du fichier en cours..."):
            # Création d'un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_option.lower()}") as tmp_file:
                file_path = tmp_file.name
                try:
                    # Exporter selon le format
                    if format_option == "RDF":
                        serialize_to_rdf(relations_corpus, file_path)
                    elif format_option == "OWL":
                        serialize_to_owl(relations_corpus, file_path)
                    
                    # Vérifier que le fichier existe et a du contenu
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="Télécharger l'ontologie",
                                data=f.read(),
                                file_name=f"ontology.{format_option.lower()}",
                                mime="application/rdf+xml"
                            )
                        st.success(f"L'ontologie a été exportée en format {format_option} avec succès !")
                    else:
                        st.error("Le fichier généré est vide ou n'a pas été créé correctement.")
                
                except Exception as e:
                    st.error(f"Erreur lors de la sérialisation : {str(e)}")
                
                finally:
                    # Supprimer le fichier temporaire
                    if os.path.exists(file_path):
                        os.remove(file_path)

if st.checkbox("Afficher le corpus pour vérification"):
    st.write(relations_corpus)