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

def train_lda_topic_model_with_mallet(contexts,path_mallet):
	dictionary=Dictionary(contexts)


def main():
	#build dictionary
	results=blacklab.search_blacklab('<s/> <s/> (<s/> containing "numb") <s/> <s/>',window=0,lemma=True)
	results=[match['complete_match'].strip() for match in results]

	for i,result in enumerate(results):
		if i==0:
			#todo filter here
			result = result.split()
			filtered_result = [word for word in result if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
			
			dct=initialize_gensim_dictionary([filtered_result])
		else:
			result = result.split()
			filtered_result = [word for word in result if ((word[0] in string.ascii_uppercase + string.ascii_lowercase))]
			add_documents_to_gensim_dictionary(dct,[filtered_result])
	
	gensim_corpus = [dct.doc2bow(bag_of_word.split()) for bag_of_word in results]
	
	lda = LdaMallet(constants.PATH_TO_MALLET, gensim_corpus, id2word=dct,num_topics=50)
	



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

