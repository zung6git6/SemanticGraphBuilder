# SemanticGraphBuilder
A semantic analysis tool that extracts concepts and relationships from text, visualizes their ontology as an interactive graph, and exports the results in RDF or OWL formats.

## How to run
```bash
streamlit run interface/🏠Home.py
```

1. Extract Concepts:
The application analyzes the provided text to extract key concepts, with the possibility to refine them. These concepts can then be used to build a semantic ontology.

2. Extract Relationships Between Concepts:
Once the concepts are extracted, the application identifies and displays semantic relationships between them, such as “is-a,” “part-of,” and similar types of relations.

3. Visualize the Ontology of Relationships:
You can explore the relationships between concepts through interactive graphs. It is also possible to visualize a semantic similarity heatmap as well as a concept dendrogram.
Additionally, synonymous and meronymous concepts are available, with adjustable threshold parameters to fine-tune these results.

4. Export the Ontology:
The application also allows you to export the ontology in various formats, such as RDF or OWL.
