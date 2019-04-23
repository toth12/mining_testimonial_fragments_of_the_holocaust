"""Print all words in a Gensim dictionary to a CVS file with."""
from utils import gensim_utils
import constants

dts = gensim_utils.load_gensim_dictionary_model(
    constants.OUTPUT_FOLDER +
    'gensimdictionary_all_words_with_phrases')

dfObj = gensim_utils.get_document_frequency_in_dictionary(
    dts, as_pandas_df=True)
# to filter the median value
# df3 = dfObj[dfObj[1] > dfObj[1].median()]
dfObj.to_csv(constants.OUTPUT_FOLDER + 'all_words_with_phrases.csv')
