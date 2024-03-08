import spacy
from transformers import pipeline
import wikipediaapi

# Ce code est utilisé pour tester la capacité des modèles à reconnaître les entités nommées dans un texte en français

# Charger le modèle français de spaCy
nlp_spacy = spacy.load("fr_core_news_sm")

# Texte pour le test
text = "Paris est la capitale de la France."

# Utiliser spaCy pour la reconnaissance d'entités nommées
print("Entités détectées par spaCy :")
doc_spacy = nlp_spacy(text)
for ent in doc_spacy.ents:
    print(ent.text, ent.label_)  # Imprimer les entités reconnues et leurs étiquettes

# Initialiser le modèle BERT pour la NER
nlp_bert_base_ner = pipeline("ner", model="dslim/bert-base-NER")

# Utiliser BERT pour la reconnaissance d'entités nommées
print("Entités détectées par BERT :")
results_bert_base = nlp_bert_base_ner(text)
for result in results_bert_base:
    print(result)  # Imprimer les entités reconnues

# Initialiser le modèle Wikineural pour la NER
nlp_wikineural_ner = pipeline("ner", model="Babelscape/wikineural-multilingual-ner")

# Utiliser Wikineural pour la reconnaissance d'entités nommées
print("Entités détectées par Wikineural :")
results_wikineural = nlp_wikineural_ner(text)
for result in results_wikineural:
    print(result)  # Imprimer les entités reconnues

# Utiliser wikipediaapi pour trouver la page Wikipedia d'une entité
wiki_wiki = wikipediaapi.Wikipedia(language='fr', user_agent="MyAppName/1.0 (myemail@example.com)")

nom_entite = "Paris"  # Nom de l'entité exemple
page_wiki = wiki_wiki.page(nom_entite)

# Vérifier si la page existe et imprimer l'URL
if page_wiki.exists():
    print("Page Wikipedia trouvée :", page_wiki.fullurl)
else:
    print("Aucune page Wikipedia trouvée pour :", nom_entite)

