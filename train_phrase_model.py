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






def main(output_filename):
	print ('Training phrase model began')
	phrase_model=gensim_utils.build_gensim_phrase_model_from_sentences(blacklab.iterable_results('<s/>',lemma=True, window = 0))	
	phrase_model.save(output_filename)
	print ('Training phrase model finished')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-o", "--output", type=str)
	


	
	args = parser.parse_args()

	if not args.output:
		raise Exception 

	
	
	main(args.output)

	#Data/Output/phrase_model

	#python3 train_phrase_model.py -o Data/Output/phrase_model_test
	