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
	if (word[0] not in string.ascii_lowercase) or (word in set(stopwords.words('english')) ):
		return utils.RULE_DISCARD

def initialize_gensim_synset_model_with_dictionary(dictionary):
	model = Word2Vec().build_vocab_from_freq(dictionary)
	return model
	

def main():
	ids=text.read_json(constants.INPUT_FOLDER+'testimony_ids.json')
	ids = [element['testimony_id'] for element in ids]
	model=build_gensim_synset_model_from_sentences(blacklab.iterable_results('<s/>',document_ids=ids,lemma=True))
	cc=sorted(model.wv.vocab.keys())
	print (len(cc))
	pdb.set_trace()
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

