from utils import gensim_utils,blacklab
import pdb
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants



def main():
	document_collection_original=blacklab.search_blacklab('<s/> <s/> (<s/> containing [lemma="naked" | lemma="undress" | lemma="strip"]) <s/> <s/>',window=0,lemma=True, include_match=True)
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



	result_lda_training = gensim_utils.train_lda_topic_model_with_mallet(document_collection_filtered,constants.PATH_TO_MALLET,3)
	result=gensim_utils.post_process_result_of_lda_topic_model(result_lda_training['model'],result_lda_training['corpus'],document_collection_original,document_collection_filtered)

	#topic_text = text.read_json('topics_texts')
	gensim_utils.write_topics_texts_to_file(result['topic_documents'],'test_output_topic_texts')

if __name__ == '__main__':
	main()