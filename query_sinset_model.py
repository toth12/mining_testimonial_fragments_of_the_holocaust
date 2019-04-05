from utils import gensim_utils,blacklab
import constants
import pdb
from gensim.models import Word2Vec

term = "naked"
model=Word2Vec.load(constants.OUTPUT_FOLDER+'test_synstets')
similar_terms=model.wv.most_similar(term,topn=30)
print (similar_terms)
pdb.set_trace()