from utils import gensim_utils,blacklab
import constants
import pdb
from gensim.models import Word2Vec

term = "father"
model=Word2Vec.load(constants.OUTPUT_FOLDER+'synsets_window_5')
similar_terms=model.wv.most_similar(term,topn=200)
print (similar_terms)
pdb.set_trace()