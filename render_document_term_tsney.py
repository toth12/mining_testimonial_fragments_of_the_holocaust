from corextopic import corextopic as ct
from sklearn.manifold import TSNE
import pandas as pd
from plotly import express as px
import plotly
import pdb
import json
from kneed import KneeLocator
import numpy as np



topic_model = ct.load("corex_topic_model")

with open('document_collection_topic_model.txt') as json_file:
    document_collection = json.load(json_file)

document_topic_matrix = topic_model.log_p_y_given_x



tsne = TSNE().fit_transform(document_topic_matrix)

principalDf = pd.DataFrame(data=tsne,
                           columns=['principal_component_1',
                                    'principal_component_2'])

principalDf['text'] = document_collection


# Plot the results of the principal component analysis without clusters
fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",hover_data=["text"])

plotly.offline.plot(fig, filename='topic_model_tsney.html')
pdb.set_trace()