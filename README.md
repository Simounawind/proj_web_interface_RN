# 🚀 Reconnaissance d'Entités Nommées avec Multiples Modèles et Interface Web 🚀

Ce projet a été développé avec passion et dévouement dans le cadre des cours "Interfaces web pour le TAL" et "Réseaux de neurones", brillamment dirigés par M. Loïc Grobol à l'Université Paris Nanterre.



## 📚 Introduction

Dans le cadre de notre exploration approfondie de l'univers de la reconnaissance d'entités nommées (NER), nous avons mis en œuvre une approche méthodique pour étudier et comparer l'efficacité de divers modèles de renom dans ce domaine. Notre quête nous a menés à sélectionner et à utiliser trois modèles pré-entraînés particulièrement distingués, chacun apportant une contribution unique à notre compréhension globale de la NER. Voici les modèles que nous avons choisis pour notre voyage :

- **spaCy** : Reconnu pour sa rapidité et son efficacité, le modèle fr_core_news_sm de spaCy offre une base solide pour la reconnaissance d'entités en français. C'est un choix privilégié pour les applications nécessitant une grande vitesse de traitement et une bonne précision.

- **dslim/bert-base-NER** : Ce modèle, basé sur l'architecture Transformer de BERT et pré-entraîné sur un vaste corpus, est spécialisé dans la reconnaissance d'entités nommées. Il est connu pour sa capacité à comprendre le contexte des mots dans des phrases, ce qui le rend extrêmement efficace pour identifier avec précision divers types d'entités.

- **Babelscape/wikineural-multilingual-ner** : Ce modèle tire parti de l'apprentissage profond pour offrir une reconnaissance d'entités nommées dans plusieurs langues, y compris le français. Grâce à son entraînement sur des données provenant de Wikipedia, il est capable de reconnaître un large éventail d'entités à travers différents domaines.

- **Fine-tuned TinyBERT** : Au-delà de l'adoption de modèles pré-entraînés réputés, nous avons également pris l'initiative ambitieuse d'adapter un modèle pour répondre précisément à nos exigences particulières. En tirant parti du [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd) , Nous avons procédé au fine-tuning d'un [modèle TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) spécifiquement pour notre projet. Le choix de TinyBERT était motivé par sa structure légère et efficace, rendant possible une haute performance tout en maintenant une consommation de ressources réduite. TinyBERT se distingue dans le domaine du TAL par sa capacité à comprimer les modèles de langue de grande taille tout en préservant leur puissance et leur efficacité. 

## 🎯 Objectif

Notre mission principale est de créer un portail web intergalactique qui permet aux utilisateurs terriens de soumettre du texte en français et d'obtenir des annotations de NER d'une précision stellaire en utilisant quatre modèles distincts : le vénérable spaCy, l'illustre BERT-base (dslim/bert-base-NER), l'énigmatique Wikineural (Babelscape/wikineural-multilingual-ner) et l'intrépide TinyBERT, finetuné avec le précieux [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd). L'objectif est d'offrir aux utilisateurs un moyen direct et visuellement intuitif de comparer les performances de chaque modèle, y compris notre adaptation personnalisée, pour répondre à des besoins spécifiques dans le domaine de la NER. Cette initiative entend proposer une ressource transparente et facilement accessible, destinée à faciliter la recherche et l'exploitation pratique de la reconnaissance d'entités nommées (NER).

## 👩‍🚀👨‍🚀 Auteurs

Cette aventure cosmique a été entreprise par Yingzi LIU et Xiaohua CUI, deux courageux étudiants de l'Université Sorbonne Nouvelle, armés de leur curiosité et de leur soif de connaissance.

## ✨ Fonctionnalités

- **Multi-modèles de Reconnaissance d'Entités Nommées (NER)**: Emploi de quatre puissants modèles de NER pour classifier et annoter le texte soumis, révélant les entités telles que les personnes, lieux, organisations, etc.
- **Interface Web**: Nous proposons une interface web élaborée et ergonomique, destinée aux utilisateurs désirant analyser des textes de manière détaillée. Ce portail permet de soumettre aisément des écrits ou documents pour une exploration NER complète.

## 📦 Contenu du Projet

1. **Modèles de NER**: Une quête épique à travers différents modèles de NER, cherchant à déterminer le plus vaillant.
2. **Fine-tuning de TinyBERT**: L'art délicat d'adapter un modèle TinyBERT avec le trésor caché qu'est le [MultiNERD dataset](https://huggingface.co/datasets/Babelscape/multinerd), dans le but d'améliorer sa performance sur des textes spécifiques.
3. **Développement d'une Interface Web**: La construction d'un portail interstellaire pour soumettre des textes et visualiser les résultats de NER, un véritable pont entre l'homme et la machine.

## Installation et exécution 🛠️

1. Clonez le dépôt GitHub sur votre navire spatial (machine locale).
2. Installez les composants nécessaires avec `pip install -r requirements.txt`.
3. Lancez le serveur web avec `python app.py`, ou via terminal avec un serveur Web comme `uvicorn`
4. Accédez à l'interface web via votre navigateur spatial pour soumettre du texte ou des fichiers et recevoir des annotations de NER dignes d'une civilisation avancée.

>  Vous devez installer un serveur Web asynchrone tel que `uvicorn`, ainsi que les modules nécessaires tels que `fastapi`. 
>
> De plus, il pourrait être nécessaire de télécharger différents modèles via `transformers`, nous vous recommandons donc fortement de télécharger ces éléments via une connexion Wi-Fi stable en raison de la taille potentielle des fichiers.

## 🤝 Contributions

Nous encourageons les contributions de tous les coins de l'univers ! Pour signaler un bug interstellaire ou proposer des améliorations cosmiques, ouvrez une issue ou soumettez une pull request.

## 🧪 Améliorations et expérimentations avec LLMs 

Notre projet est en constante évolution, surtout à l'ère des grands modèles de langage. Nous avons expérimenté avec des LLMs plus petits, tels que `LLama2` et `Qwen1.5`. Notamment, `Qwen1.5`, un grand modèle de langage produit en Chine, offre un modèle de 0.5B, se caractérisant par sa compacité et son efficacité. Cependant, en raison des besoins spécifiques de notre projet et des problèmes de compatibilité avec notre système de développement sur Mac, nous avons dû abandonner cette option.

N'hésitez pas à contribuer et à nous faire part de vos suggestions d'amélioration ! 🌟

## 📞 Contact

Pour toute question ou pour transmettre des signaux de détresse, contactez-nous à :

- Yingzi LIU : yingzi.liu@sorbonne-nouvelle.fr
- Xiaohua CUI : xiaohua.cui@sorbonne-nouvelle.fr

Nous vous remercions infiniment pour votre intérêt dans notre projet et nous espérons que vous y trouverez autant de plaisir et d'émerveillement que nous en avons eu à le développer !

## 📜 Licence

Ce projet est distribué sous la licence MIT, un pacte de bonne volonté intergalactique. Pour plus de détails, consultez le fichier LICENSE.



