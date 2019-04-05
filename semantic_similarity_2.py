import tensorflow as tf
import tensorflow_hub as hub
import os
import pdb
import numpy as np
from sklearn.decomposition import PCA
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot
import scipy
from sklearn.manifold import TSNE
from utils import blacklab
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#nlp = spacy.load('en_core_web_md')
#text represents our raw text document

query = '<s/> containing ([lemma="naked" | lemma="undress" | lemma="strip"])'
document_collection_original=blacklab.search_blacklab(query,window=0,lemma=False, include_match=True)

sentences = [sentence['complete_match'] for sentence in document_collection_original if len (sentence['complete_match']) > 100]




url = "/Users/gmt28/Documents/Workspace/mining_testimonial_fragments_of_the_holocaust/Bin/9bb74bc86f9caffc8c47dd7b33ec4bb354d9602d"

embed = hub.Module(url)
result = []
embeddings_list = []
# This tells the model to run through the 'sentences' list and return the default output (1024 dimension sentence vectors).
for i in range(0,len(sentences), 10):
	print (i)
	sentence = sentences[i:i+100] 
	
	embeddings = embed(
	    sentence,
	    signature="default",
	    as_dict=True)["default"]
	embeddings_list.append(embeddings)

	#Start a session and run ELMo to return the embeddings in variable x
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  for f,embeddings in enumerate(embeddings_list):
  	print (f)
  	x = sess.run(embeddings)
  	result.append(x)
  sess.close()

pdb.set_trace()
print ('PCA begins')

pca = PCA(n_components=50) #reduce down to 50 dim
y = pca.fit_transform(x)

print ('TSNE begins')
y = TSNE(n_components=2).fit_transform(y) # further reduce to 2 dim using t-SNE


data = [
    go.Scatter(
        x=[i[0] for i in y],
        y=[i[1] for i in y],
        mode='markers',
        text=[i for i in sentences],
    marker=dict(
        size=16,
        color = [len(i) for i in sentences], #set color equal to a variable
        opacity= 0.8,
        colorscale='Viridis',
        showscale=False
    )
    )
]
layout = go.Layout()
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )
fig = go.Figure(data=data, layout=layout)
file = plot(fig, filename='Sentence encode.html')
