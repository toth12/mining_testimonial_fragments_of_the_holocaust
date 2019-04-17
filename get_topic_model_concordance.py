from utils import gensim_utils,blacklab,text
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import sys, getopt
import argparse
import inspect
from collections import Counter
import difflib







def main(query,output_filename,terms_to_remove=[],window=0,topicn=50):
	print ('Training lda model began')
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	query_parameters = [(i, values[i]) for i in args]
	document_collection_original=blacklab.search_blacklab(query,window=window,lemma=True, include_match=True)
	
	#Eliminate overlapping results
	document_collection_original_for_updating = document_collection_original
	
	#find those ids that are present more than once
	ids={i['testimony_id']:[]for i in document_collection_original}
	
	for element in document_collection_original:
		ids[element['testimony_id']].append(element)

	for element in ids:
		if len(ids[element])>0:
			for i in range(0,len(ids[element])-1):
				
				match_element_1=ids[element][i]['complete_match']

		#Iterate through and try to find similar matches
				for text in ids[element][i+1:]:
					match_element_2=text['complete_match']
					diff = difflib.SequenceMatcher(None, match_element_1,match_element_2).ratio()
					if diff > 0.75:
						try:
							document_collection_original.pop(document_collection_original.index(text))
						except:
							pass
	
			
	#Eliminate search terms
	terms_to_remove = terms_to_remove + constants.DATA_SPECIFIC_STOP_WORDS
	#get rid of stop words
	document_collection_filtered_without_search_terms = []
	for text in document_collection_original:
		new_text = []
		for word in text['complete_match'].split():
			if (word.lower() not in terms_to_remove):
				new_text.append(word)
		document_collection_filtered_without_search_terms.append(' '.join(new_text))
	


	#use the phraser model
	phraser_model = Phraser(Phrases.load(constants.OUTPUT_FOLDER+'phrase_model'))
	document_collection=[' '.join(phraser_model[match.strip().split()]) for match in document_collection_filtered_without_search_terms]
	ids= [match['testimony_id'] for match in document_collection_original]

	
	#get rid of stop words
	document_collection_filtered = []
	for text in document_collection:
		new_text = []
		for word in text.split():
			if (word.lower() not in set(stopwords.words('english')) and (word[0] in string.ascii_uppercase + string.ascii_lowercase)):
				new_text.append(word)
		document_collection_filtered.append(' '.join(new_text))


	
	result_lda_training = gensim_utils.train_lda_topic_model_with_mallet(document_collection_filtered,constants.PATH_TO_MALLET,[],topicn)
	result=gensim_utils.post_process_result_of_lda_topic_model(result_lda_training['model'],result_lda_training['corpus'],document_collection_original,document_collection_filtered)

	#topic_text = text.read_json('topics_texts')
	gensim_utils.write_topics_texts_to_file(result['topic_documents'],constants.OUTPUT_FOLDER+output_filename,query_parameters = query_parameters)
	print ('Training lda model finished')
if __name__ == '__main__':

	#open the CSV file


	parser = argparse.ArgumentParser()
	
	parser.add_argument("-s", "--synset", type=str)
	parser.add_argument("-q", "--query", type=str)
	parser.add_argument("-w", "--window", type=int)
	parser.add_argument("-o", "--output", type=str)
	parser.add_argument("-topicn", type=int)



	#parser.add_argument('window')
	args = parser.parse_args()

	if not args.output:
		raise Exception 

	if not args.topicn:
		args.topicn = 50

	if not args.window:
		args.window = 0

	if args.synset:
		input_data=text.ReadCSVasDict('Data/Input/Synsets/synsets.csv')

		synonyms=[element['Synonyms'].strip() for element in input_data if element['Covering_Term']==args.synset]
		query = []

		for element in synonyms[0].split(','):
			if '_' in element:
				result = '('
				element = element.split('_')
				for e in element:
					result = result + '[word="'+e+'"]'
				result=result+')'
				query.append(result)
			else:
				query.append('[lemma="'+element.strip()+'"]')
		query = '|'.join(query)
		query = '[]{10}('+query+ ')([ ]{10})'
		 
		
		
		main(query,args.output,synonyms[0].split(','),args.window,args.topicn)
	else:
		main(args.query,args.output,[],args.window,args.topicn)

	