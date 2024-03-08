from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import spacy
import wikipediaapi
import aiofiles

# Créer une instance d'application FastAPI
app = FastAPI()

# Monter le répertoire statique pour servir des fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialiser les templates Jinja2 pour la génération de HTML
templates = Jinja2Templates(directory="templates")

# Charger le modèle spaCy français pour la reconnaissance d'entités nommées (NER)
nlp_spacy = spacy.load("fr_core_news_sm")

# Initialiser les modèles de reconnaissance d'entités nommées (NER) de Transformers
nlp_bert_base_ner = pipeline("ner", model="dslim/bert-base-NER")
nlp_wikineural_ner = pipeline("ner", model="Babelscape/wikineural-multilingual-ner")

# Initialiser le modèle TinyBERT finetuned pour la NER
tokenizer_custom = AutoTokenizer.from_pretrained("Tinybert-finetuned-ner")
model_custom = AutoModelForTokenClassification.from_pretrained("Tinybert-finetuned-ner")
nlp_custom = pipeline("ner", model=model_custom, tokenizer=tokenizer_custom)

# Initialiser WikipediaAPI avec un agent utilisateur personnalisé pour les requêtes
wiki_wiki = wikipediaapi.Wikipedia(language='fr', user_agent="MyAppName/1.0 (myemail@example.com)")

app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
async def root(request: Request):
    # Générer et retourner la page d'accueil en utilisant un template Jinja2
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze/")
async def analyze_text(text: str = Form(default=""), file: UploadFile = File(None)):
    try:
        # Gérer la réception du contenu texte ou d'un fichier
        if file:
            contents = await file.read()
            text = contents.decode("utf-8")
            await file.close()
            print("Contenu du fichier reçu :", text)
        else:
            print("Contenu texte reçu :", text)

        if not text:
            return {"error": "Aucun texte fourni"}

        # Traitement du texte avec spaCy pour la reconnaissance d'entités nommées
        entities_spacy = []
        doc_spacy = nlp_spacy(text)
        print("Entités détectées par spaCy :", doc_spacy.ents)
        for ent in doc_spacy.ents:
            wiki_page = wiki_wiki.page(ent.text)
            url = wiki_page.fullurl if wiki_page.exists() else None
            entities_spacy.append({"entity": ent.text, "type": ent.label_, "url": url})

        print("Entités traitées par spaCy :", entities_spacy)
        
        # Traitement du texte avec le modèle BERT-base-NER de Transformers
        entities_bert_base_ner = []
        results_bert_base = nlp_bert_base_ner(text)
        print("Sortie du modèle BERT base NER :", results_bert_base)
        for result in results_bert_base:
            entity = text[result['start']:result['end']]
            wiki_page = wiki_wiki.page(entity)
            url = wiki_page.fullurl if wiki_page.exists() else None
            score = float(result['score']) if 'score' in result else None
            entities_bert_base_ner.append({"entity": entity, "type": result['entity'], "score": score, "url": url})

        print("Entités traitées par BERT :", entities_bert_base_ner)

        # Traitement du texte avec le modèle Wikineural-multilingual-ner de Transformers
        entities_wikineural_ner = []
        results_wikineural = nlp_wikineural_ner(text)
        print("Sortie du modèle Wikineural NER :", results_wikineural)
        for result in results_wikineural:
            entity = text[result['start']:result['end']]
            wiki_page = wiki_wiki.page(entity)
            url = wiki_page.fullurl if wiki_page.exists() else None
            score = float(result['score']) if 'score' in result else None
            entities_wikineural_ner.append({"entity": entity, "type": result['entity'], "score": score, "url": url})

        print("Entités traitées par Wikineural :", entities_wikineural_ner)

        # utiliser le modèle TinyBERT finetuned pour la NER
        results_custom = nlp_custom(text)

        entities_custom = []
        for result in results_custom:
            entity_text = text[result['start']:result['end']]
            score = result.get('score')
            if score is not None:  # faire en sorte que le score existe
                score = float(score)  # convertir le score en float standard
            entities_custom.append({"entity": entity_text, "type": result['entity'], "score": score})

        # retourner la réponse JSON
        # return JSONResponse(content={"entities_custom_model": entities_custom})

        # Fusionner les résultats et les retourner
        return {
            "entities_spacy": entities_spacy,
            "entities_bert_base_ner": entities_bert_base_ner,
            "entities_wikineural_ner": entities_wikineural_ner,
            "entities_custom_model": entities_custom
        }
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return {"error": "Erreur interne du serveur"}, 500

