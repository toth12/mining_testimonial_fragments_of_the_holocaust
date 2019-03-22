from utils import blacklab
from utils import gensim_utils
import pdb
from utils import text
import constants
import pandas as pd	



ids=text.read_json(constants.INPUT_FOLDER+'testimony_ids.json')

#Verbs
for i,element in enumerate(ids):

	results=blacklab.search_blacklab('[pos="V.*"]',window=0,lemma=True,document_id=element['testimony_id'])
	verbs=[[match['complete_match'].strip() for match in results]]
	
	print ('Verbs')
	print (i)
	if i==0:
		dct=gensim_utils.initialize_gensim_dictionary(verbs)
	else:
		gensim_utils.add_documents_to_gensim_dictionary(dct,verbs)
dct.save(constants.OUTPUT_FOLDER+'gensimdictionary_all_verbs')

dts=gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER+'gensimdictionary_all_verbs')
dts.filter_extremes(no_below=10, no_above=0.95)
dfObj=gensim_utils.get_document_frequency_in_dictionary(dts,as_pandas_df=True)
df3 = dfObj[dfObj[1] > dfObj[1].median()]
df3.to_csv(constants.OUTPUT_FOLDER+'all_verbs_filtered_no_below_10_no_above_95_percent.csv')

#Adjectives


for i,element in enumerate(ids):

	results=blacklab.search_blacklab('[pos="JJ.*"]',window=0,lemma=True,document_id=element['testimony_id'])
	adjectives=[[match['complete_match'].strip() for match in results]]
	print ('Adjectives')
	print (i)
	if i==0:
		dct=gensim_utils.initialize_gensim_dictionary(adjectives)
	else:
		gensim_utils.add_documents_to_gensim_dictionary(dct,adjectives)
dct.save(constants.OUTPUT_FOLDER+'gensimdictionary_all_adjectives')

dts=gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER+'gensimdictionary_all_adjectives')
dts.filter_extremes(no_below=10, no_above=0.95)
dfObj=gensim_utils.get_document_frequency_in_dictionary(dts,as_pandas_df=True)
df3 = dfObj[dfObj[1] > dfObj[1].median()]
df3.to_csv(constants.OUTPUT_FOLDER+'all_adjectives_filtered_no_below_10_no_above_95_percent.csv')



#Nouns


for i,element in enumerate(ids):

	results=blacklab.search_blacklab('[pos="NN.*"]',window=0,lemma=True,document_id=element['testimony_id'])
	nouns=[[match['complete_match'].strip() for match in results]]
	print ('Nouns')
	print (i)
	if i==0:
		dct=gensim_utils.initialize_gensim_dictionary(nouns)
	else:
		gensim_utils.add_documents_to_gensim_dictionary(dct,nouns)
dct.save(constants.OUTPUT_FOLDER+'gensimdictionary_all_nouns')

dts=gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER+'gensimdictionary_all_adjectives')
dts.filter_extremes(no_below=10, no_above=0.95)
dfObj=gensim_utils.get_document_frequency_in_dictionary(dts,as_pandas_df=True)
df3 = dfObj[dfObj[1] > dfObj[1].median()]
df3.to_csv(constants.OUTPUT_FOLDER+'all_nouns_filtered_no_below_10_no_above_95_percent.csv')

gensim_utils.main()





