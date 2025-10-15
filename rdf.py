from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, OWL, RDFS

def serialize_to_owl(relations_corpus, output_file="ontology.owl"):
    # Créer un graphe RDF
    g = Graph()
    
    # Définir les namespaces
    EX = Namespace("http://example.org/")
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf", RDF)
    
    # Ajouter la déclaration de l'ontologie
    ontology_uri = URIRef(EX["MyOntology"])
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    
    # Ensemble pour suivre les propriétés et individus déjà déclarés
    declared_properties = set()
    declared_individuals = set()
    
    # Parcourir le dictionnaire
    for sentence, triplets in relations_corpus.items():
        for triplet in triplets:
            subject, relation, obj = triplet
            # Créer des URI pour chaque élément
            subj_uri = URIRef(EX[subject.replace(" ", "_")])
            pred_uri = URIRef(EX[relation.replace(" ", "_")])
            
            # Vérifier si l'objet est un littéral ou un individu
            if " " in obj or len(obj.split()) > 1:  # Si c'est une phrase
                obj_node = Literal(obj)
            else:  # Si c'est un seul mot
                obj_node = URIRef(EX[obj.replace(" ", "_")])
            
            # Déclarer le sujet comme individu
            if subj_uri not in declared_individuals:
                g.add((subj_uri, RDF.type, OWL.NamedIndividual))
                declared_individuals.add(subj_uri)
            
            # Déclarer l'objet comme individu si ce n'est pas un littéral
            if isinstance(obj_node, URIRef) and obj_node not in declared_individuals:
                g.add((obj_node, RDF.type, OWL.NamedIndividual))
                declared_individuals.add(obj_node)
            
            # Déclarer la propriété
            if pred_uri not in declared_properties:
                # Si l'objet est un littéral, c'est une DataProperty, sinon ObjectProperty
                if isinstance(obj_node, Literal):
                    g.add((pred_uri, RDF.type, OWL.DatatypeProperty))
                else:
                    g.add((pred_uri, RDF.type, OWL.ObjectProperty))
                declared_properties.add(pred_uri)
            
            # Ajouter le triplet au graphe
            g.add((subj_uri, pred_uri, obj_node))
            
            # Ajouter la propriété mentionedIn comme AnnotationProperty
            mentioned_in_uri = EX["mentionedIn"]
            if mentioned_in_uri not in declared_properties:
                g.add((mentioned_in_uri, RDF.type, OWL.AnnotationProperty))
                declared_properties.add(mentioned_in_uri)
            g.add((subj_uri, mentioned_in_uri, Literal(sentence)))

    # Sérialiser le graphe en RDF/XML (format OWL)
    try:
        g.serialize(destination=output_file, format="xml")
        print(f"Fichier OWL créé avec succès : {output_file}")
    except Exception as e:
        print(f"Erreur lors de la sérialisation : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de dictionnaire
    sample_corpus = {
        "Le chat mange la souris": [
            ("chat", "mange", "souris"),
            ("Cat", "mange", "Mouse")
        ],
        "Marie aime lire des livres": [
            ("Marie", "aime", "lire"),
            ("Marie", "lit", "livres")
        ]
    }
    
    # Appeler la fonction
    serialize_to_owl(sample_corpus, "ontology.owl")