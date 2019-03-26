from utils import gensim_utils,blacklab
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import sys, getopt
import argparse
import inspect
from utils import gensim_utils, text, blacklab




#This is not working without a phrase model, it is to be improved

def main(output_filename,path_to_phrase_model,window=5):

	#Load the gensim dictionary
	dts=gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER+'gensimdictionary_all_words_with_phrases')
	#Apply filter here
	dts.filter_extremes(no_below=10, no_above=0.95)
	#Initialize an empty model with the dictionary above
	model = gensim_utils.initialize_gensim_synset_model_with_dictionary(dts,window = window)
	#Read the ids 
	ids=text.read_json(constants.INPUT_FOLDER+'testimony_ids.json')

	#All tokens except punctuation

	for i,element in enumerate(ids):
	if path_to_phrase_model:
			sentences=blacklab.iterable_results('<s/>',lemma=True, path_to_phrase_model=constants.OUTPUT_FOLDER+"phrase_model",window = 0,document_ids=[element['testimony_id']])
			sentences = list(sentences)
			model.train(sentences,epochs=model.epochs,total_examples=model.corpus_count)
		else:
			model=gensim_utils.build_gensim_synset_model_from_sentences(blacklab.iterable_results('<s/>',lemma=True,window = window))

	cc=sorted(model.wv.vocab.keys())
	model.save(constants.OUTPUT_FOLDER+output_filename)
	
	print (len(cc))
	
	print ('Training synset model finished')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-w", "--window", type=int)
	parser.add_argument("-o", "--output", type=str)
	parser.add_argument("-ppm", "--path_to_phrase_model", type=str)
	


	
	args = parser.parse_args()

	if not args.output:
		raise Exception 

	if not args.window:
		args.window = 5
	
	
	main(args.output,args.path_to_phrase_model,args.window)

	#Data/Output/phrase_model

	#python3 train_gensim_synset_model.py -o synset_model_test -ppm Data/Output/phrase_model_test
	