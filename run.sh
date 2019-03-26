#!/bin/sh
python3 train_phrase_model.py -o Data/Output/phrase_model_test
python3 get_document_frequencies.py  
python3 train_gensim_synset_model.py -o test_synstets -ppm Data/Output/phrase_model_test