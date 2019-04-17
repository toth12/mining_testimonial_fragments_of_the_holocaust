from utils import gensim_utils,blacklab,text
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import sys, getopt
import argparse
import inspect




def main(query,output_filename,window=0):
	print ('Training lda model began')
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	query_parameters = [(i, values[i]) for i in args]
	x=blacklab.search_blacklab(query,window=window,lemma=False, include_match=True)
	
	
	
	
	output_text = ''
	
	output_text = output_text + query+'\n'
	for i,result in enumerate(x):
		
		output_text = output_text + str(i)+'. '+result['testimony_id']+'\n'
		output_text = output_text + result['complete_match']+'\n'
	
	f = open(output_filename,'w')
	f.write(output_text)
	f.close()


if __name__ == '__main__':

	#open the CSV file


	parser = argparse.ArgumentParser()
	
	parser.add_argument("-1", "--synset1", type=str)
	parser.add_argument("-2", "--synset2", type=str)
	parser.add_argument("-w", "--window", type=int)
	parser.add_argument("-o", "--output", type=str)
	



	#parser.add_argument('window')
	args = parser.parse_args()

	if not args.output:
		raise Exception
	if not args.synset1:
		raise Exception

	if not args.synset2:
		raise Exception
	
	if not args.window:
		args.window = 50

	
	input_data=text.ReadCSVasDict('Data/Input/Synsets/synsets.csv')

	synonyms_1=[element['Synonyms'].strip() for element in input_data if element['Covering_Term']==args.synset1]
	query_1 = []
	query_2 = []

	for element in synonyms_1[0].split(','):
		if '_' in element:
			result = '('
			element = element.split('_')
			for e in element:
				result = result + '[lemma="'+e+'"]'
			result=result+')'
			query_1.append(result)
		else:
			query_1.append('[lemma="'+element.strip()+'"]')
	query_1 = '('+'|'.join(query_1)+')'

	synonyms_2=[element['Synonyms'].strip() for element in input_data if element['Covering_Term']==args.synset2]

	for element in synonyms_2[0].split(','):
		if '_' in element:
			result = '('
			element = element.split('_')
			for e in element:
				result = result + '[lemma="'+e+'"]'
			result=result+')'
			query_2.append(result)
		else:
			query_2.append('[lemma="'+element.strip()+'"]')

	query_2 = '('+'|'.join(query_2)+')'

	new_query = '('+ query_1 + '[]{0,15}' + query_2 + ')|'+'('+ query_2 + '[]{0,15}' + query_1 + ')'
	pdb.set_trace()
		 
		
		
	main(new_query,args.output,args.window)
	
	