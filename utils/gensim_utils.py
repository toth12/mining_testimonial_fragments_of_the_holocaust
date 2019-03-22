'''Functions to use Gensim'''
from gensim.corpora import Dictionary
import pdb
import pandas as pd
from gensim.models import Word2Vec
from utils import blacklab, text
from gensim import utils
import string
from nltk.corpus import stopwords
import constants
from gensim.models.phrases import Phrases, Phraser
from itertools import chain
from gensim.models.wrappers import LdaMallet
import numpy as np
from gensim.models import CoherenceModel
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim



def initialize_gensim_dictionary(text):
	dct = Dictionary(text)	
	return dct

def add_documents_to_gensim_dictionary(gensim_dictionary_model,text):
	gensim_dictionary_model.add_documents(text)

def get_gensim_dictionary(gensim_dictionary_model):
	return list(gensim_dictionary_model.token2id.keys())

def get_document_frequency_in_dictionary(gensim_dictionary_model,as_pandas_df=False):
	look_up=gensim_dictionary_model.dfs
	dictionary=get_gensim_dictionary(gensim_dictionary_model)
	result={ }
	for i,word in enumerate(dictionary):
		result[word]=look_up[i]

	if as_pandas_df==True:
		dfObj = pd.DataFrame(result.items())
		return dfObj
	else:
		return result

def load_gensim_dictionary_model(path_to_gensim_dictionary_model):
	dct=Dictionary.load(path_to_gensim_dictionary_model)
	return dct

def build_gensim_synset_model_from_sentences(sentences):
	model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4,trim_rule=trim_rule)
	return model

def trim_rule(word,count,min_count):
	if (word[0] not in string.ascii_uppercase + string.ascii_lowercase) or (word in set(stopwords.words('english')) ):
		return utils.RULE_DISCARD

def initialize_gensim_synset_model_with_dictionary(dictionary):
	model = Word2Vec().build_vocab_from_freq(dictionary)
	return model

def find_similar_terms(term,path_to_model,n=10):
	model=Word2Vec.load(path_to_model)
	similar_terms=model.wv.most_similar(term,topn=n)
	return similar_terms

def build_gensim_phrase_model_from_sentences(sentences):
	phrases = Phrases(sentences, min_count=5, threshold=2)
	return phrases

def identify_phrases(sentence,path_to_gensim_phrase_model):
	phrase_model = Phrases.load(path_to_gensim_phrase_model)
	phraser_model=Phraser(phrase_model)
	new_sentence=phraser_model[sentence]
	return new_sentence

def train_lda_topic_model_with_mallet(texts,path_mallet, num_topics=50, scoring = False, start = 2 , step = 3):
	preprocessed_corpus = []
	for i,text in enumerate(texts):
		if i==0:
			#todo filter here
			text = text.split()
			filtered_text = [word for word in text if (word[0] in string.ascii_uppercase + string.ascii_lowercase)]
			filtered_text = [word for word in filtered_text if (word not in set(stopwords.words('english')))]
			preprocessed_corpus.append(filtered_text)
			dct=initialize_gensim_dictionary([filtered_text])
		else:
			text = text.split()
			filtered_text = [word for word in text if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
			filtered_text = [word for word in filtered_text if (word not in set(stopwords.words('english')))]
			preprocessed_corpus.append(filtered_text)
			add_documents_to_gensim_dictionary(dct,[filtered_text])
	
	gensim_corpus = [dct.doc2bow(bag_of_word.split()) for bag_of_word in texts]
	if scoring:

		coherence_values = []
    	
		for n in range(start, num_topics, step):
			lda = LdaMallet(constants.PATH_TO_MALLET, gensim_corpus, id2word=dct, num_topics=n)
			coherencemodel = CoherenceModel(model=lda, texts=preprocessed_corpus, dictionary=dct, coherence='c_v')
			coherence_values.append(coherencemodel.get_coherence())

		return coherence_values
    	

	else:
		lda = LdaMallet(constants.PATH_TO_MALLET, gensim_corpus, id2word=dct,num_topics=num_topics)
		return {'model':lda,'corpus' : gensim_corpus}


def post_process_result_of_lda_topic_model(lda_model,gensim_corpus,document_collection):
	#Prepare containers to store results
	#Container to keep the document topic matrix
	document_topic_matrix=[]
	#Container to keep topics and the closest texts to each topic
	topic_closest_doc_with_topics_words = []
	#Container to keep topics
	all_topics = lda_model.show_topics(50)

	#Create an LDA corpus from the original gensim corpus
	lda_corpus = lda_model[gensim_corpus]

	#Iterate through the lda corpus and create the document topic matrix
	for i,documents in enumerate(lda_corpus):
		#Data returned is not proper numpy matrix
		document_topic_matrix.append(np.array([elements[1] for elements in documents]))

	#Create the proper numpy matrix
	document_topic_matrix=np.vstack(document_topic_matrix)

	#Find the closest texts to a given topic
	#Iterate through the transpose of the document topic matrix
	for i,element in enumerate(document_topic_matrix.T):
		#Identify the id of 15 closest texts of each topic
		closest=element.argsort(axis=0)[-15:][::-1]
		#Create a container to keep each text with the id above
		texts = []
		for element in closest:
			texts.append(document_collection[element])
		#Append them to container
		topic_closest_doc_with_topics_words.append({'texts':texts,'topic_words':all_topics[i]})
	return {'topic_documents':topic_closest_doc_with_topics_words , 'document_topic_matrix' : document_topic_matrix}
	


def main():
	#todo: elimination of searched terms should happen later, eliminate stop words
	document_collection=blacklab.search_blacklab('<s/><s/> <s/> (<s/> containing [lemma="naked" | lemma="undress" | lemma="strip"]) <s/> <s/> <s/>',window=0,lemma=True, include_match=True)
	document_collection=[match['complete_match'].strip() for match in document_collection]

	#result_lda_training = train_lda_topic_model_with_mallet(document_collection,constants.PATH_TO_MALLET,50)
	result_lda_training = train_lda_topic_model_with_mallet(document_collection,constants.PATH_TO_MALLET,20,True)

	limit=20; start=2; step=3;
	x = range(start, limit, step)
	plt.plot(x, result_lda_training)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
	plt.savefig('demo.pdf')

	'''
	text.write_json('corpus', model['corpus'])
	model['model'].save('ldamodel')
	text.write_json('plain_texts', results)

	post_processed = post_process_result_of_lda_topic_model(result_lda_training['model'],result_lda_training['corpus'],document_collection)
	'''

	topics_texts = text.read_json('topics_texts')
	
	output_text = ''
	for i,element in enumerate(topics_texts):
		topic_words =  ' '.join([str(topic) for topic in element['topic_words']])
		output_text = output_text + str(i) + '. '+ topic_words +':' + '\n\n'
		for document in element['texts']:
			output_text = output_text + document + '\n\n\n'
		output_text = output_text + '-----------------------\n\n'
	f = open('output.txt','w')
	f.write(output_text)
	f.close()
	pdb.set_trace()
		
	