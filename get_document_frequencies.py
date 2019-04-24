"""Print document frequency of different word types."""
from utils import blacklab
from utils import gensim_utils
from utils import text
import constants


def main():
    """Extract different type of tokens and their document frequency.

    Extract all tokens(without punctuation),all nouns, all verbs and all
    adjectives from the corpus and print them with document frequency
    to CVS files.

    """
    print('Getting document frequencies began')
    ids = text.read_json(constants.INPUT_FOLDER + 'testimony_ids.json')

    # All tokens except punctuation
    for i, element in enumerate(ids):

        results = blacklab.iterable_results('<s/>',
                                            lemma=True,
                                            path_to_phrase_model=constants.
                                            OUTPUT_FOLDER + "phrase_model",
                                            document_ids=[element
                                                            ['testimony_id']],
                                            window=0)
        """Get all words from the document and
        represent each document as a bag of words
        """
        all_words = [[word for sentence in results for word in sentence]]

        if i == 0:
            dct = gensim_utils.initialize_gensim_dictionary(all_words)
        else:
            gensim_utils.add_documents_to_gensim_dictionary(dct, all_words)

    dct.save(constants.OUTPUT_FOLDER + 'gensimdictionary_all_words_with_phrases')
    # if one wants to filter them
    dts = gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER +
                                                    'gensimdictionary_all_words_with_phrases')
    dts.filter_extremes(no_below=10, no_above=0.95)

    dfobj = gensim_utils.get_document_frequency_in_dictionary(dts, as_pandas_df=True)
    # df3 = dfObj[dfObj[1] > dfObj[1].median()]
    dfobj.to_csv(constants.OUTPUT_FOLDER + 'all_words_with_phrases.csv')
    dts.save(constants.OUTPUT_FOLDER + 'gensimdictionary_all_words_with_phrases_filtered_no_below_10_no_above_095')

    # Verbs
    for i, element in enumerate(ids):

        results = blacklab.search_blacklab('[pos="V.*"]', window=0, lemma=True,
                                           document_id=element['testimony_id'])
        verbs = [[match['complete_match'].strip() for match in results]]

        if i == 0:
            dct = gensim_utils.initialize_gensim_dictionary(verbs)
        else:
            gensim_utils.add_documents_to_gensim_dictionary(dct, verbs)
    dct.save(constants.OUTPUT_FOLDER + 'gensimdictionary_all_verbs')

    dts = gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER +
                                                    'gensimdictionary_all_verbs')
    dts.filter_extremes(no_below=10, no_above=0.95)
    dfobj = gensim_utils.get_document_frequency_in_dictionary(dts,
                                                              as_pandas_df=True
                                                              )
    # df3 = dfObj[dfObj[1] > dfObj[1].median()]
    dfobj.to_csv(constants.OUTPUT_FOLDER +
                 'all_verbs_filtered_no_below_10_no_above_95_percent_above.csv')

    # Adjectives

    for i, element in enumerate(ids):

        results = blacklab.search_blacklab('[pos="JJ.*"]', window=0,
                                           lemma=True,
                                           document_id=element['testimony_id'])
        adjectives = [[match['complete_match'].strip() for match in results]]

        if i == 0:
            dct = gensim_utils.initialize_gensim_dictionary(adjectives)
        else:
            gensim_utils.add_documents_to_gensim_dictionary(dct, adjectives)
    dct.save(constants.OUTPUT_FOLDER + 'gensimdictionary_all_adjectives')

    dts = gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER +
                                                    'gensimdictionary_all_adjectives')
    dts.filter_extremes(no_below=10, no_above=0.95)
    dfobj = gensim_utils.get_document_frequency_in_dictionary(dts,
                                                              as_pandas_df=True)
    # df3 = dfObj[dfObj[1] > dfObj[1].median()]
    dfobj.to_csv(constants.OUTPUT_FOLDER + 'all_adjectives_filtered_no_below_10_no_above_95_percent.csv')

    # Nouns

    for i, element in enumerate(ids):

        results = blacklab.search_blacklab('[pos="NN.*"]', window=0,
                                           lemma=True,
                                           document_id=element['testimony_id'])
        nouns = [[match['complete_match'].strip() for match in results]]

        if i == 0:
            dct = gensim_utils.initialize_gensim_dictionary(nouns)
        else:
            gensim_utils.add_documents_to_gensim_dictionary(dct, nouns)
    dct.save(constants.OUTPUT_FOLDER + 'gensimdictionary_all_nouns')

    dts = gensim_utils.load_gensim_dictionary_model(constants.OUTPUT_FOLDER +
                                                    'gensimdictionary_all_nouns'
                                                    )
    dts.filter_extremes(no_below=10, no_above=0.95)
    dfobj = gensim_utils.get_document_frequency_in_dictionary(dts,
                                                              as_pandas_df=True
                                                              )
    # df3 = dfObj[dfObj[1] > dfObj[1].median()]
    dfobj.to_csv(constants.OUTPUT_FOLDER + 'all_nouns_filtered_no_below_10_no_above_95_percent.csv')
    print('Getting document frequencies finished')


if __name__ == '__main__':
    main()
