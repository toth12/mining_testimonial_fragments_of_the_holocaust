from utils import blacklab
from utils import gensim_utils
import pdb	

ids=['HVT-1','HVT-2','HVT-3','HVT-4']

for i,element in enumerate(ids):
	results=blacklab.search_blacklab('[pos="V.*"]',window=0,lemma=True,document_id=element)
	verbs=[[match['complete_match'].strip() for match in results]]
	
	if i==0:
		dct=gensim_utils.initialize_gensim_dictionary(verbs)
	else:
		gensim_utils.add_documents_to_gensim_dictionary(dct,verbs)
	dct.filter_extremes(no_above=0.5,no_below=1)
	print (gensim_utils.get_document_frequency_in_dictionary(dct))