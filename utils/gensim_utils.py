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

def train_lda_topic_model_with_mallet(texts,path_mallet,num_topics=50):

	for i,text in enumerate(texts):
		if i==0:
			#todo filter here
			text = text.split()
			filtered_text = [word for word in text if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
			
			dct=initialize_gensim_dictionary([filtered_text])
		else:
			text = text.split()
			filtered_text = [word for word in text if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
			add_documents_to_gensim_dictionary(dct,[filtered_text])
	
	gensim_corpus = [dct.doc2bow(bag_of_word.split()) for bag_of_word in texts]
	
	lda = LdaMallet(constants.PATH_TO_MALLET, gensim_corpus, id2word=dct,num_topics=50)
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

	pdb.set_trace()




def main():
	#build dictionary
	#todo: elimination of searched terms should happen later
	'''results=blacklab.search_blacklab('<s/> <s/> (<s/> containing [lemma="naked" | lemma="undress" | lemma="strip"]) <s/> <s/>',window=0,lemma=True, include_match=True)
	results=[match['complete_match'].strip() for match in results]
	
	model = train_lda_topic_model_with_mallet(results,constants.PATH_TO_MALLET,50)
	text.write_json('corpus', model['corpus'])
	model['model'].save('ldamodel')
	text.write_json('plain_texts', results)'''

	post_process_result_of_lda_topic_model(LdaMallet.load('ldamodel'),text.read_json('corpus'),text.read_json('plain_texts'))
	pdb.set_trace()
	result = {'model': LdaMallet.load('ldamodel'),'corpus': text.read_json('corpus')}
	plain_text = text.read_json('plain_texts')
	lda_corpus = result['model'][result['corpus']]
	all_topics = result['model'].show_topics(50)
	
	#prepare container
	#create a container to keep the document topic matrix
	document_topic_matrix=[]
	#prepare a container for keeping the topic and the documents for which the given topic is the first one
	topic_first_doc = {v:[] for v in range(0,50)}

	#prepare a container for topics and the closest texts to them
	topic_closest_doc = []

	topic_closest_doc_with_topics_words = {}

	#end of containers


	for i,documents in enumerate(lda_corpus):
		
		#find the strongest topic for document
		most_important_topic_for_document=np.array([elements[1] for elements in documents]).argmax()

		

		
		#make a proper np array to keep the the document per topic relationship and add it to the document matrix
		document_topic_matrix.append(np.array([elements[1] for elements in documents]))
		
		topic_first_doc[most_important_topic_for_document].append(plain_text[i])

		'''

		#add the id of the topic which is the strongest for a given document 
		topics_list.append(topics)

		#add the sentence and id to the topic which is the strongest
		

		'''
		
	#create the final document term matrix in numpy
	document_topic_matrix=np.vstack(document_topic_matrix)

	#Find the closest texts to a given topic
	for i,element in enumerate(document_topic_matrix.T):
		
		closest=element.argsort(axis=0)[-15:][::-1]
		texts = []
		for element in closest:
			texts.append(plain_text[element])
		
		topic_closest_doc.append({'texts':texts,'topic_words':all_topics[i]}) 


		


	pdb.set_trace()
	
	#idaig

	all_words = ''.join(results).split()
	filtered_results = [word for word in all_words if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
    #todo: filter out stopwords	
	dct=initialize_gensim_dictionary([filtered_results])
	pdb.set_trace()
	
	print ('Adjectives')
	print (i)
	


	train_lda_topic_model_with_mallet(contexts,'somepath')

	'''
	ids=text.read_json(constants.INPUT_FOLDER+'testimony_ids.json')
	ids = [element['testimony_id'] for element in ids][0:16]
	phrase_model=build_gensim_phrase_model_from_sentences(blacklab.iterable_results('<s/>',document_ids=ids,lemma=True))	
	phrase_model.save(constants.OUTPUT_FOLDER+"phrase_model")

	
	model=build_gensim_synset_model_from_sentences(blacklab.iterable_results('<s/>',document_ids=ids,lemma=True, path_to_phrase_model=constants.OUTPUT_FOLDER+"phrase_model"))
	cc=sorted(model.wv.vocab.keys())
	model.save(constants.OUTPUT_FOLDER+'word2vecmodel')
	print (len(cc))
	pdb.set_trace()
	
	'''
	'''
	model=Word2Vec.build_vocab(vocabulary)



	pdb.set_trace()
	dct=initialize_gensim_dictionary(text_1+text_2)
	corpus = [dct.doc2bow(text) for text in text_1+text_2]
	word_freq_dic=[[(dct[id], freq) for id, freq in cp] for cp in corpus]
	pdb.set_trace()
	model=initialize_gensim_synset_model_with_dictionary(dct)
	
	dct=initialize_gensim_dictionary(text_1)
	add_documents_to_gensim_dictionary(dct,text_2)
	get_document_frequency_in_dictionary(dct)
	print (dct.token2id)
	pdb.set_trace()'''

