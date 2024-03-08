#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning d'un modèle TinyBERT pour NER (la reconnaissance d'entités nommées) en français en utilisant le dataset MultiNERD(https://github.com/Babelscape/multinerd)

# In[15]:


get_ipython().system(' pip install datasets transformers seqeval')
get_ipython().system(' pip install transformers[torch]')
get_ipython().system(' pip install accelerate -U')


# ## Importation des bibliothèques

# In[16]:


from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np


# ## Chargement du dataset MultiNERD et filtrage des données françaises

# In[17]:


# Définir le ratio du sous-ensemble
subset_ratio = 0.1

def select_subset(dataset, ratio):
    """
    Sélectionne un sous-ensemble d'un dataset.
    
    Paramètres :
    dataset : Le dataset à traiter.
    ratio : Le ratio du dataset à sélectionner.
    
    Retourne :
    Un sous-ensemble du dataset.
    """
    # Mélanger aléatoirement le dataset
    dataset = dataset.shuffle(seed=42)
    # Calculer la taille du sous-ensemble selon le ratio spécifié
    subset_size = int(len(dataset) * ratio)
    # Retourner le sous-ensemble du dataset
    return dataset.select(range(subset_size))

# Charger les datasets
datasets = load_dataset("Babelscape/multinerd")

# Filtrer pour ne conserver que les données en français
train_dataset = datasets["train"].filter(lambda exemple: exemple['lang'] == "fr")
val_dataset = datasets["validation"].filter(lambda exemple: exemple['lang'] == "fr")
test_dataset = datasets["test"].filter(lambda exemple: exemple['lang'] == "fr")

# Sélectionner un sous-ensemble pour les datasets d'entraînement, de validation et de test
train_dataset = select_subset(train_dataset, subset_ratio)
val_dataset = select_subset(val_dataset, subset_ratio)
test_dataset = select_subset(test_dataset, subset_ratio)


# ## Définition des étiquettes et chargement du modèle TinyBERT

# In[18]:


# Création d'un dictionnaire pour les étiquettes de la reconnaissance d'entités nommées
labels_vocab = {
    "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6,
    "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9, "I-BIO": 10, "B-CEL": 11, "I-CEL": 12,
    "B-DIS": 13, "I-DIS": 14, "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18,
    "B-INST": 19, "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24,
    "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, "I-VEHI": 30,
}
label_list = list(labels_vocab.keys())
labels_vocab_reverse = {v: k for k, v in labels_vocab.items()}

# Chargement du tokenizer et du modèle pré-entraîné TinyBERT
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForTokenClassification.from_pretrained(
    "prajjwal1/bert-tiny", 
    num_labels=len(label_list), 
    label2id=labels_vocab, 
    id2label=labels_vocab_reverse
)


# ## Prétraitement et alignement des étiquettes

# In[19]:


def tokenize_and_align_labels(examples):
    # Tokenize les exemples
    tokenized_inputs = tokenizer(examples["tokens"], 
                                 truncation=True, 
                                 padding="max_length", 
                                 max_length=128,  # Taille maximale des tokens
                                 is_split_into_words=True,
                                 return_tensors='pt')
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)
    
    # Ajoute les étiquettes dans les exemples
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



# ## Mappage des ensembles de données

# In[20]:


train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
val_tokenized = val_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)


# ## Configuration des paramètres d'entraînement

# In[21]:


training_args = TrainingArguments(
    "TinyBERT-finetuned-ner", # Répertoire de sortie
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=10,  # Optimisation pour les GPUs/CPUs avec peu de mémoire
)


# ## Calcul des métriques

# In[22]:


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval") # Chargement de la métrique seqeval
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ## Entraînement et évaluation

# In[23]:


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate(test_tokenized)


# ## Utilisation du modèle TinyBERT entraîné pour faire des prédictions

# In[24]:


# Utilisation du modèle entraîné pour faire des prédictions
predictions, labels, _ = trainer.predict(test_tokenized)
predictions = np.argmax(predictions, axis=2)

# Suppression des indices ignorés (pour les tokens spéciaux)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# Calcul des indicateurs de performance
metric = load_metric("seqeval")
results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)


# ## Sauvegarde du modèle et du tokenizer

# In[25]:


# Sauvegarde du modèle
model.save_pretrained("Tinybert-finetuned-ner")

# Sauvegarde du tokenizer
tokenizer.save_pretrained("Tinybert-finetuned-ner")


# ## Test

# In[29]:


from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Charger le modèle et le tokenizer
tokenizer = AutoTokenizer.from_pretrained("Tinybert-finetuned-ner")
model = AutoModelForTokenClassification.from_pretrained("Tinybert-finetuned-ner")

def predict(texts, tokenizer, model, labels_vocab_reverse):
    # Tokenize les textes
    tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # Faire une prédiction
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)

    # Aligner les prédictions avec les tokens
    predicted_labels = []
    for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
        # Convertir les ids en tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        prediction_indices = predictions[i].tolist()
        
        # Aligner les tokens avec les prédictions
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Récupérer les ids des tokens
        aligned_labels = [labels_vocab_reverse[pred_idx] for token, pred_idx, word_id in zip(tokens, prediction_indices, word_ids) if word_id is not None and token not in tokenizer.all_special_tokens]

        predicted_labels.append(aligned_labels)

    return predicted_labels

# Textes à prédire
texts = ["Paris et France", "Je aime bien Paris"]

# Faire des prédictions
predicted_labels = predict(texts, tokenizer, model, labels_vocab_reverse)
for text, labels in zip(texts, predicted_labels):
    print("Text:", text)
    print("Labels:", labels)


