
# coding: utf-8

# # node2vec
# ---
# [node2vec](http://snap.stanford.edu/node2vec/) for link prediction:
# 1. Perform train-test split
# 1. Train skip-gram model on random walks within training graph
# 2. Get node embeddings from skip-gram model
# 3. Create bootstrapped edge embeddings by taking the Hadamard product of node embeddings
# 4. Train a logistic regression classifier on these edge embeddings (possible edge --> edge score between 0-1)
# 5. Evaluate these edge embeddings on the validation and test edge sets
# 
# node2vec source code: https://github.com/aditya-grover/node2vec

# ## 1. Read in Graph Data

# In[2]:

import networkx as nx
#import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
#import pickle


# In[5]:

#EGO_USER = 0 # which ego network to look at

# Load pickled (adj, feat) tuple
#network_dir = './fb-processed/{0}-adj-feat.pkl'.format(EGO_USER)
#with open(network_dir, 'rb') as f:
#    adj, features = pickle.load(f)
    
#g = nx.Graph(adj) # re-create graph using node indices (0 to num_nodes-1)


# In[93]:

Gr = nx.read_edgelist("ca-GrQc.mtx", nodetype=int, delimiter=" ")
for edge in Gr.edges():
    Gr[edge[0]][edge[1]]['weight'] = 1    


# In[92]:




# In[ ]:

#print Gr.number_of_edges()
#print Gr.number_of_nodes()


# In[52]:

# draw network
# nx.draw_networkx(Gr, with_labels=False, node_size=50, node_color='r')
# plt.show()


# ## 2. Preprocessing/Train-Test Split

# In[ ]:

from gae.preprocessing import mask_test_edges
np.random.seed(0) # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(Gr)



# In[ ]:

# Perform train-test split
adj_train, train_edges, train_edges_false, val_edges, val_edges_false,     test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.005, val_frac=.001, prevent_disconnect= True)
g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges


# In[67]:

# Inspect train/test split
#print "Total nodes:", adj_sparse.shape[0]
#print "Total edges:", int(adj_sparse.nnz/2) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
#print "Training edges (positive):", len(train_edges)
#print "Training edges (negative):", len(train_edges_false)
#print "Validation edges (positive):", len(val_edges)
#print "Validation edges (negative):", len(val_edges_false)
#print "Test edges (positive):", len(test_edges)
#print "Test edges (negative):", len(test_edges_false)

output_training_test = open("output_split.txt", "w")
output_training_test.write(("\n Total nodes:" + str(adj_sparse.shape[0])))
output_training_test.write("\n Total edges:" + str(int(adj_sparse.nnz/2)))
output_training_test.write("\n Training edges (positive): " + str(len(train_edges))) 
output_training_test.write("\n Training edges (negative): " + str(len(train_edges_false)))
output_training_test.write("\n Validation edges (positive): " + str(len(val_edges)))
output_training_test.write("\n Validation edges (negative): " + str(len(val_edges_false)))
output_training_test.write("\n Test edges (positive): " + str(len(test_edges)))
output_training_test.write("\n Test edges (negative): " + str(len(test_edges_false)))

output_training_test.close()
# ## 3. Train node2vec (Learn Node Embeddings)

# In[68]:

import node2vec
from gensim.models import Word2Vec


# In[69]:

# node2vec settings
# NOTE: When p = q = 1, this is equivalent to DeepWalk

P = 1 # Return hyperparameter
Q = 1 # In-out hyperparameter
WINDOW_SIZE = 10 # Context size for optimization
NUM_WALKS = 10 # Number of walks per source
WALK_LENGTH = 80 # Length of walk per source
DIMENSIONS = 128 # Embedding dimension
DIRECTED = False # Graph directed/undirected
WORKERS = 8 # Num. parallel workers
ITER = 1 # SGD epochs


# In[70]:

# Preprocessing, generate walks
g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
g_n2v.preprocess_transition_probs()
walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walks = [map(str, walk) for walk in walks]

# Train skip-gram model
model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

# Store embeddings mapping
emb_mappings = model.wv


# ## 4. Create Edge Embeddings

# In[71]:

# Create node embeddings matrix (rows = nodes, columns = embedding features)
emb_list = []
for node_index in range(0, adj_sparse.shape[0]):
    node_str = str(node_index)
    node_emb = emb_mappings[node_str]
    emb_list.append(node_emb)
emb_matrix = np.vstack(emb_list)


# In[72]:

# Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs


# In[73]:

# Train-set edge embeddings
pos_train_edge_embs = get_edge_embeddings(train_edges)
neg_train_edge_embs = get_edge_embeddings(train_edges_false)
train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
#
## Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])
#
## Val-set edge embeddings, labels
pos_val_edge_embs = get_edge_embeddings(val_edges)
neg_val_edge_embs = get_edge_embeddings(val_edges_false)
val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])
#
# Test-set edge embeddings, labels
pos_test_edge_embs = get_edge_embeddings(test_edges)
neg_test_edge_embs = get_edge_embeddings(test_edges_false)
test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
#
## Create val-set edge labels: 1 = real edge, 0 = false edge
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])
#
#
## ## 5. Evaluate Edge Embeddings
#
## In[74]:
#
## Train logistic regression classifier on train-set edge embeddings
from sklearn.linear_model import LogisticRegression
edge_classifier = LogisticRegression(random_state=0)
edge_classifier.fit(train_edge_embs, train_edge_labels)
#
#
## In[75]:
#
## Predicted edge scores: probability of being of class "1" (real edge)
val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
val_roc = roc_auc_score(val_edge_labels, val_preds)
val_ap = average_precision_score(val_edge_labels, val_preds)
#
#
## In[76]:
#
## Predicted edge scores: probability of being of class "1" (real edge)
test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
test_roc = roc_auc_score(test_edge_labels, test_preds)
test_ap = average_precision_score(test_edge_labels, test_preds)
#
#
## In[77]:
#

result_file = open("result_file.txt", "w")

result_file.write("node2vec Validation ROC score: " + str(val_roc))
result_file.write(" \n node2vec Validation AP score: " + str(val_ap))
result_file.write(" \n node2vec Test ROC score: " + str(test_roc))
result_file.write(" \n node2vec Test AP score: " + str(test_ap))

result_file.close()

#print 'node2vec Validation ROC score: ', str(val_roc)
#print 'node2vec Validation AP score: ', str(val_ap)
#print 'node2vec Test ROC score: ', str(test_roc)
#print 'node2vec Test AP score: ', str(test_ap)
#
#
## In[ ]:
#
#
#
