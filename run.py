from utils import blacklab
from utils import gensim_utils
import pdb
from utils import text	

ids=['HVT-1','HVT-2','HVT-3','HVT-4']
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
dct.save('gensimdic')

