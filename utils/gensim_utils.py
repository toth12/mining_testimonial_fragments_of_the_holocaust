'''Functions to use Gensim'''
from gensim.corpora import Dictionary
import pdb


def initialize_gensim_dictionary(text):
	dct = Dictionary(text)	
	return dct

def add_documents_to_gensim_dictionary(gensim_dictionary_model,text):
	gensim_dictionary_model.add_documents(text)

def get_gensim_dictionary(gensim_dictionary_model):
	return list(gensim_dictionary_model.token2id.keys())

def get_document_frequency_in_dictionary(gensim_dictionary_model):
	look_up=gensim_dictionary_model.dfs
	dictionary=get_gensim_dictionary(gensim_dictionary_model)
	result={ }
	for i,word in enumerate(dictionary):
		result[word]=look_up[i]

	return result
	

def main():
	text_1 = [['human', 'interface', 'cat']]
	text_2= [["cat", "say", "meow"]]
	dct=initialize_gensim_dictionary(text_1)
	add_documents_to_gensim_dictionary(dct,text_2)
	get_document_frequency_in_dictionary(dct)
	print (dct.token2id)
	pdb.set_trace()

