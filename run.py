from utils import blacklab
from utils import gensim_utils
import pdb
from utils import text
import constants
import pandas as pd	
'''
ids=text.read_json('testimony_ids.json')

for i,element in enumerate(ids):

	results=blacklab.search_blacklab('[pos="V.*"]',window=0,lemma=True,document_id=element['testimony_id'])
	verbs=[[match['complete_match'].strip() for match in results]]
	
	print (i)
	if i==0:
		dct=gensim_utils.initialize_gensim_dictionary(verbs)
	else:
		gensim_utils.add_documents_to_gensim_dictionary(dct,verbs)
#dct.filter_extremes(no_below=134, no_above=0.9)
'''
dts=gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER+'gensimdictionary_all_verbs')
dts.filter_extremes(no_below=10, no_above=0.95)
dictionary=gensim_utils.get_document_frequency_in_dictionary(dts)

dfObj = pd.DataFrame(dictionary.items())

df3 = dfObj[dfObj[1] > dfObj[1].median()]

df3.to_csv(constants.OUTPUT_FOLDER+'all_verbs_4.csv')
pdb.set_trace()
print(dfObj[1].median())




