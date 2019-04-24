#!/bin/sh
python3 train_phrase_model.py -o Data/Output/phrase_model
python3 get_document_frequencies.py  
python3 train_gensim_synset_model.py -o synsets_window_5 -ppm Data/Output/phrase_model
python3 train_gensim_synset_model.py -w 10 -o synsets_window_10 -ppm Data/Output/phrase_model
python3 train_gensim_synset_model.py -w 15 -o synsets_window_15 -ppm Data/Output/phrase_model