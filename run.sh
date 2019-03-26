#!/bin/sh

python3 get_document_frequencies.py || python3 train_phrase_model.py -o Data/Output/phrase_model_test || python3 train_gensim_synset_model.py -o test_synstets -ppm Data/Output/phrase_model_test || python3 get_topic_model_concordance.py --query='<s/> <s/> (<s/> containing [lemma="naked" | lemma="undress" | lemma="strip"]) <s/> <s/>' -o topic_model_concordance -w 100 -topicn 10