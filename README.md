# Mining Testimonial Fragments of the Holocaust

The goal of this project is to reconstruct the pieces of the collective experience (aka testimonial fragments) by victims of the Holocaust. Since the data underlying this project is not yet available to the general public, running the project will be possible only at a later stage. 

## General Description

This project reconstructs the pieces of the collective experience by identifying and extracting recurrent physical and emotional experiences from a corpus of 2681 testimonies with survivors. Computationally, the goal is to retrieve short chunks of texts the meaning of which is very similar by implemmenting a bottom-up or let-the-data speak approach. For instance, the following two sentences convey the same experience:

* "They would beat them to undress and shoot them and bury them -- and then had to bury them."
* "Because about, they were all, they made them undress and they were shooting them and they were falling into the graves."

The retrieval of sentences conveying similar meaning is still in initial phase. No ready-to-use general purpose alghorithmic solution is available. Hence, to resolve this problem, this project applies various approaches and combines them into a pipeline that also involves human supervision. This repository contains the computational implementation of each step of the pipeline:

* Finding recurrent terms by computing document frequencies
* Identifying synonyms of recurrent terms and forming synonym sets with a standard word-to-vector model
* Identification of multiword expressions
* Finding all occurrences of synonym sets in the data set with the help of Corpus Query Language
* Identifying recurrent patterns underlying the occurrences of synonym sets with the help of Latent Dirichlet Allocation
* Human investigation of recurrent patterns with the purpose of identifying sentences that feature semantic similarity
* Transformation of these sententences into testimonial fragments (short extracts that capture a given experience)

The retrieved testimonial fragments are published on a purpose-built data edition of Holocaust testimonies named "Let them speak." This repository also contains methods to construct the input data to the "Let them speak" ecosystem.

## The Data Set

The data set consists of the transcript of 2681 oral history interviews with survivors of the Holocaust, approximately 60 million tokens. Each interview had earlier undergone a standard linguistic processing with the help of Stanford Parser: sentence splitting, tokenization, pos tagging and lemmatization. The data set is available as a linguistic corpus; it is searchable by means of BlackLab, a corpus engine by Dutch Language Institute (http://inl.github.io/BlackLab/blacklab-server-overview.html). Many methods in this repo use Blacklab as backend to obtain data; they communicate with the engine through HTPP requests.




## Prerequisites and Installation of Libraries and Packages

As backend this repository uses the BlackLab engine. Hence this has to be installed and the dataset needs to be available through the engine. The API endpoint is recorded in constants.py. First test if Blacklab is up and running by querying the API endpoint with curl.

```
curl -i http://localhost:8080/blacklab-server-1.7.1/lts/ 
```

If blacklab is not available, this is returning an error message "Failed to connect."

The project also uses Mallet topic modelling. Download and install it as described here http://mallet.cs.umass.edu/download.php. First create a Bin folder to store Mallet and install it into the Bin folder. Path to Mallet is recorded in constants.py

```
mkdir Bin
```

The project requirements can be installed with conda into a virtual environment named "base" by running the following command in the project folder (conda needs to be preinstalled):
```
conda create --name base --file requirements.txt
```

Then activate the virtual environment from the project folder:
```
conda activate base 
```

The post processing of testimonial fragments requires a running Stanford Core NLP server with an endpoint http://localhost:9000. Install a Stanford Core NLP server from here: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html and run it on port 9000.

The repository by default does not contain folders to store input and output data. (Data folders are ignored by git, see .gitignore). Before running any of the processes run the following commands in the main project folder to create data folders and copy the necessary input data to there:

```
mkdir -p Data/Output/TensorBoard & mkdir -p Data/Input/Synsets & mkdir -p Data/Input/Fragments 
```

```
cp synsets.csv Data/Input/Synsets/
```

```
cp testimony_ids.json Data/Input/
```

```
cp testimonial_fragments_old.xlsx Data/Input/Fragments/
```

```
cp testimonial_fragments.xlsx Data/Input/Fragments/
```
## Pipeline Steps

### Train a gensim phrase model to identify multiword expressions

Since all subsequent steps use this model, it is crucial to train this model first.

```
python3 train_phrase_model.py -o Data/Output/phrase_model
```

This method identifies multiword expressions in the data by iterating through all sentences of the corpus. The method uses lemmas and not words as input.

### Compute the document frequency of certain word groups

```
python3 get_document_frequencies.py 
``` 

To identify recurrence in the data, as a first step lemmas above the median document frequency are idenfied. This method computes the document frequency of all lemmas identified by Stanford Parser (i.e. not words!) (multiword expressions included), all adjectives (multiword expressions excluded), all verbs (multiword expressions excluded), all nouns (multiword expressions excluded). The method first obtains specific word groups from each document of the corpus and creates a gensim dictionary. Each gensim dictionary is pruned; terms present in less than 10 documents or more than 95% of all documents are excluded; median document frequency is calculated after excluding these extremes. Original - not pruned -  gensim dictionary models are saved into Data/Output:
* gensimdictionary_all_verbs
* gensimdictionary_all_words_with_phrases
* gensimdictionary_all_adjectives
* gensimdictionary_all_nouns

Final results (document frequency of different types of lemmas) are saved into Data/Output as CSV files:

* all_words_with_phrases.csv
* all_verbs_filtered_no_below_10_no_above_95_percent_above.csv
* all_adjectives_filtered_no_below_10_no_above_95_percent.csv
* all_nouns_filtered_no_below_10_no_above_95_percent.csv

### Investigate those terms that are above the median document frequency

This is a step to be accomplished by a human investigator who identifies terms above the median document frequency describing emotional and physical experiences in different word groups. Terms such as "shake", "fear", "run" are preselected at this stage.

### Train a synset model that helps to find the synonyms of the term preselected in the previous step. 

A synset model with different windows are trained; lemmas are used and the training process uses the phrase model trained above, as well as the gensim dictionary model (gensimdictionary_all_words_with_phrases). The vocabulary is also pruned before the training process: terms that are present in more than 95% of the documents or less than 25 documents are eliminated before the training of the synset model begins. Throughout the training process all sentences from each document are extracted and passed to the model. Results of the training process are copied to Data/Output folder.

```
python3 train_gensim_synset_model.py -o synsets_window_1 -w 1 -ppm Data/Output/phrase_model
```

```
python3 train_gensim_synset_model.py -o synsets_window_2 -w 2 -ppm Data/Output/phrase_model
```

```
python3 train_gensim_synset_model.py -o synsets_window_3 -w 3 -ppm Data/Output/phrase_model
```

```
python3 train_gensim_synset_model.py -o synsets_window_5 -ppm Data/Output/phrase_model
```

```
python3 train_gensim_synset_model.py -o synsets_window_10 -w 10 -ppm Data/Output/phrase_model
```

```
python3 train_gensim_synset_model.py -o synsets_window_15 -w 15 -ppm Data/Output/phrase_model
```

### Render clusters of word embeddings with tensorboard

Utility function to render synonym clusters with t-SNEY. 

```
python3 render_synset_model_tensor_flow.py
```

### Query the synset models

Synset models can be queried: 

```
python3 query_synset_model.py -t "kill"
```
By default, the function returns the first 50 closest terms and uses the model with window five. This can be overwritten with -n and -d parameters.

```
python3 query_synset_model.py -t "kill" -d 3 -n 150
```

If model with specific window is not available, an error is thrown.

### Form synsets and record them in Data/Input/Synsets/synsets.csv

Elements of synsets is to be recorded in the CSV file in the column 'Synonyms' as comma separated list. Later this can be used as input.

### Find patterns (combination of topic words) underlying the occurrences of a synset in the corpus:

The method in get_topic_model_concordance.py uses Mallet LDA implementation to get the topic words. First all occurrences of a synset or a CQL pattern are retrieved (only lemmas and not words!); overlapping search results are eliminated with difflib. Search results undergo a number of preprocessing steps before the LDA training. The search terms used in the query are eliminated (if they are not eliminated, they will be the main keywords that LDA reproduces); similarly stopwords are removed, as well as those terms that are not beginning with letters of the alphabet (numbers for instance). Each search result is also passed to the phrase model so that multiword expressions can be also included. Next a gensim dictionary is created from each document that are represented as bag of words (no weighting is used). Extremes are filtered out: terms that do not occurr at least in 10 search results or that occurr in more than 90% of documents are removed. From the dictionary and from the filtered document collection, a gensim corpus object is constructed; this is the input of the Gensim MALLET LDA model. The result of training (Gensim Mallet LDA model and Gensim Corpus Model) are passed to a function of postprocessing. This gets each topic and the n closest documents to it and prints it to the output file. The following command collects all occurrences of the word "numb" and the five words before and after the occurrence (-w 5), creates an LDA model from them and prints results into the file 'numbLDA.'

```
python3 get_topic_model_concordance.py -q '["numb"]' -o numbLDA -w 5
```
The following command retrieves all occurrences of the synset naked, (elements of the synset retrieved from Data/Input/Synsets/synsets.csv) and writes the results to Data/Output/nakednessLDA. At the moment the ten terms preceding and following the occurrences are taken into consideration when training the LDA model.

```
python3 get_topic_model_concordance.py -s 'naked' -o nakednessLDA
```

### Insert manually fragments into Data/Input/Fragments/testimonial_fragments.xlsv

### Find computationally the sentence id of each fragment and merge with existing fragments (testimonial_fragments_old.xlsv). Stanford Corenlp server must be running on localhost:9000. Result is testimonial_fragments_updated.csv

```
python3 update_fragments.py
```
### Create the input for the Let them Speak ecosystem; this results in Data/Output/testimonial_fragments_updated.json

```
python3 transform_fragments_in_csv_to_json_for_fragments_collection.py
```

## Development Road Map:

* Implementation of non negative matrix factorization to find topic words
* Implementation of affinity propagation clustering to find prototypical instances
* Implementation of ELMO with Tensorflow (see commit 36ec78069e4cd0)
* Implementation of a deep learning model to find context words of a given term




