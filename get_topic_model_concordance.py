from utils import gensim_utils,blacklab
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import sys, getopt
import argparse
import inspect






def main(query,output_filename,window=50,topicn=50):
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	query_parameters = [(i, values[i]) for i in args]
	document_collection_original=blacklab.search_blacklab(query,window=window,lemma=True, include_match=True)
	#use the phraser model
	phraser_model = Phraser(Phrases.load(constants.OUTPUT_FOLDER+'phrase_model'))
	document_collection=[' '.join(phraser_model[match['complete_match'].strip().split()]) for match in document_collection_original]

	#get rid of stop words
	document_collection_filtered = []
	for text in document_collection:
		new_text = []
		for word in text.split():
			if (word not in set(stopwords.words('english')) and (word[0] in string.ascii_uppercase + string.ascii_lowercase)):
				new_text.append(word)
		document_collection_filtered.append(' '.join(new_text))



	result_lda_training = gensim_utils.train_lda_topic_model_with_mallet(document_collection_filtered,constants.PATH_TO_MALLET,topicn)
	result=gensim_utils.post_process_result_of_lda_topic_model(result_lda_training['model'],result_lda_training['corpus'],document_collection_original,document_collection_filtered)

	#topic_text = text.read_json('topics_texts')
	gensim_utils.write_topics_texts_to_file(result['topic_documents'],constants.OUTPUT_FOLDER+output_filename,query_parameters = query_parameters)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
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
		args.window = 50
	
	
	main(args.query,args.output,args.window,args.topicn)
	