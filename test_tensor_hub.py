import tensorflow as tf
import tensorflow_hub as hub
import os
os.environ['TFHUB_CACHE_DIR'] = '/Users/gmt28/Documents/Workspace/mining_testimonial_fragments_of_the_holocaust/bin'

with tf.Graph().as_default():
  module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
  embed = hub.Module(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
