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






def main(output_filename,path_to_phrase_model,window=5):
	#ids=text.read_json(constants.INPUT_FOLDER+'testimony_ids.json')
	#ids = [element['testimony_id'] for element in ids][0:16]
	if path_to_phrase_model:
		model=gensim_utils.build_gensim_synset_model_from_sentences(blacklab.iterable_results('<s/>',lemma=True, path_to_phrase_model=constants.OUTPUT_FOLDER+"phrase_model"),window = window)
	
	else:
		model=gensim_utils.build_gensim_synset_model_from_sentences(blacklab.iterable_results('<s/>',lemma=True,window = window))

	cc=sorted(model.wv.vocab.keys())
	model.save(constants.OUTPUT_FOLDER+output_filename)
	
	print (len(cc))
	

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
	
	
	main(args.output,args.window,args.path_to_phrase_model)

	#Data/Output/phrase_model

	#python3 train_gensim_synset_model.py -o synset_model_test -ppm Data/Output/phrase_model_test
	