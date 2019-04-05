import tensorflow as tf
import tensorflow_hub as hub
import os
import pdb
import numpy as np
from sklearn.decomposition import PCA
import scipy
from sklearn.manifold import TSNE

'''os.environ['TFHUB_CACHE_DIR'] = '/Users/gmt28/Documents/Workspace/mining_testimonial_fragments_of_the_holocaust/bin'

elmo_model = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)

pdb.set_trace()

embeddings = elmo_model(["the cat is on the mat", "dogs are in the fog"], signature="default",as_dict=True)["elmo"]
'''

sentences = [['she','is','nice','.'],['she','is','cute','.'],['she','is','not','ugly','.']]

sentences = ['she is nice .'],['she is cute']
for sentence in sentences:
	#words = tf.constant(sentence)

	#words = tf.reshape(words, [-1])


	url = "/Users/gmt28/Documents/Workspace/mining_testimonial_fragments_of_the_holocaust/Bin/9bb74bc86f9caffc8c47dd7b33ec4bb354d9602d"
	embed = hub.Module(url)


	embeddings = embed(sentences, signature="default", as_dict=True)['default']


	#Start a session and run ELMo to return the embeddings in variable x
	with tf.Session() as sess:
	  sess.run(tf.global_variables_initializer())
	  sess.run(tf.tables_initializer())
	  x = sess.run(embeddings)
	  results.append(x)

#scipy.spatial.distance.cosine(results[0]['elmo'],results[2]['elmo'])
pdb.set_trace()
pca = PCA(n_components=50) #reduce down to 50 dim
y = pca.fit_transform(results)

y = TSNE(n_components=2).fit_transform(y) # further reduce to 2 dim using t-SNE