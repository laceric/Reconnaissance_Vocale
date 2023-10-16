# Reconnaissance_Vocale
Projet Fil Rouge de la formation professionnelle Data Scientist de l'organisme Data Scientest.

Data :
Les données utilisées viennent du dossier : train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )
De l'ensemble LibriSpeech ASR corpus (Identifier: SLR12 )
lien : https://www.openslr.org/12

Ce jeu de donnée rassemble 100 heures d'audio en anglais avec leur transcription et des méta-data.

Fichiers avec transcription et méta-data :
- BOOKS_data
- CHAPTERS_data
- SPEAKERS_data
(Ces 3 fichiers sont construit à partir des fichiers du dossier : train-clean-100.tar.gz)

Fichiers audios :
Les fichiers audio (6.3 Go) sont a télécharger via la page suivante : https://www.openslr.org/12

Code :
Le code est regroupé en 3 fichiers, deux modules avec des fonctions personnalisées et un adapté pour tourner sur Google Colab.
- Code_GC : fichier python correspondant à la page Google Colab
- Module_reco_voc : fichier correspondant au premier module lié au préprocessing
- Module_reco_voc_2 : fichier correspondant au second module lié au modèle et à l'entrainement et analyse des résultats

Rapport :
- Rapport : Le rapport du projet sous format pdf.

Streamlit :
- Streamlit : Partie Streamlit pour la présentation du projet.
