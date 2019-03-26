from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import pdb
from gensim.models.nmf import Nmf


# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

#Train the model on the corpus.
nmf = Nmf(common_corpus, num_topics=10)

pdb.set_trace()