from utils import gensim_utils,blacklab
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import sys, getopt
import argparse
import inspect
from gensim.models.nmf import Nmf






def main(query,output_filename,window=50,topicn=50):
	print ('Training nmf model began')
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	query_parameters = [(i, values[i]) for i in args]
	document_collection_original=blacklab.search_blacklab(query,window=window,lemma=True, include_match=False)
	print ("Search finished")
	document_collection=[match['complete_match'].strip() for match in document_collection_original[0:100]]

	#Use the phraser model
	
	phraser_model = Phraser(Phrases.load(constants.OUTPUT_FOLDER+'phrase_model'))
	document_collection=[' '.join(phraser_model[match['complete_match'].strip().split()]) for match in document_collection_original]
	print ("Phraser model done")
	#get rid of stop words
	document_collection_filtered = document_collection
	'''
	for text in document_collection:
		new_text = []
		for word in text.split():
			if (word not in set(stopwords.words('english')) and (word[0] in string.ascii_uppercase + string.ascii_lowercase)):
				new_text.append(word)
		document_collection_filtered.append(' '.join(new_text))
	'''
	print ("Filtering done")
	
	#build the corpus
	preprocessed_corpus = []

	for i,text in enumerate(document_collection_filtered):
		if i==0:
			print (i)
			text = text.split()
			
			
			dct=gensim_utils.initialize_gensim_dictionary([text])
		else:
			print (i)
			text = text.split()
			gensim_utils.add_documents_to_gensim_dictionary(dct,[text])
	#Filter it here
	
	dct.filter_extremes(no_below=10, no_above=0.95)
	
	gensim_corpus = [dct.doc2bow(bag_of_word.split()) for bag_of_word in document_collection_filtered]
	
	#text = document_collection_filtered[0].split()
	nmf = Nmf(gensim_corpus, num_topics=50)
	words = list(dct.token2id.keys())

	topics =  nmf.print_topics(50)
	for topic in topics:

		topic_words = topic[1].split('+')
		print_topic = []
		for topic_word in topic_words:
			print_topic.append(words[int(topic_word.split('*')[1][1:].strip()[:-1])])
		print (' '.join(print_topic))

	#get topic of a given document: nmf.get_document_topics(gensim_corpus[0])
	#dct.token2id.keys()
	#nmf.show_topic(10)
	#nmf.get_document_topics(dct.doc2bow(preprocessed_corpus[0]))
	pdb.set_trace()


	#result_lda_training = gensim_utils.train_lda_topic_model_with_mallet(document_collection_filtered,constants.PATH_TO_MALLET,topicn)
	#result=gensim_utils.post_process_result_of_lda_topic_model(result_lda_training['model'],result_lda_training['corpus'],document_collection_original,document_collection_filtered)

	#topic_text = text.read_json('topics_texts')
	#gensim_utils.write_topics_texts_to_file(result['topic_documents'],constants.OUTPUT_FOLDER+output_filename,query_parameters = query_parameters)
	#print ('Training lda model finished')
if __name__ == '__main__':
	query = '[".*"]{50}[lemma="naked" | lemma="undress" | lemma="strip" | lemma="nude"] [".*"]{50}'
	output = 'something'
	main(query,output,window = 0)
	'''parser = argparse.ArgumentParser()
	
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
	
	
	main(args.query,args.output,args.window,args.topicn)'''


	