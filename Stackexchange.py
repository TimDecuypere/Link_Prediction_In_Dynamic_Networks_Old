
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

# 

# ## 1. Read in Graph Data

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
#import pickle


# In[2]:


#EGO_USER = 0 # which ego network to look at

# Load pickled (adj, feat) tuple
#network_dir = './fb-processed/{0}-adj-feat.pkl'.format(EGO_USER)
#with open(network_dir, 'rb') as f:
#    adj, features = pickle.load(f)
    
#g = nx.Graph(adj) # re-create graph using node indices (0 to num_nodes-1)


# Step 1: Load in the networks, make the training and test edge lists

# In[4]:


MasterGraph = nx.read_edgelist("Stack-exch/stack.csv", nodetype=int, delimiter=",")
for edge in MasterGraph.edges():
    MasterGraph[edge[0]][edge[1]]['weight'] = 1

print MasterGraph.number_of_nodes()
print MasterGraph.number_of_edges()

G1 = nx.read_edgelist("Stack-exch/stack_p1.csv", nodetype = int, delimiter = ",")
for edge in G1.edges():
    G1[edge[0]][edge[1]]['weight'] = 1
G2 = nx.read_edgelist("Stack-exch/stack_p2.csv", nodetype = int, delimiter = ",")
for edge in G2.edges():
    G2[edge[0]][edge[1]]['weight'] = 1
G3 = nx.read_edgelist("Stack-exch/stack_p3.csv", nodetype = int, delimiter = ",")
for edge in G3.edges():
    G3[edge[0]][edge[1]]['weight'] = 1
G4 = nx.read_edgelist("Stack-exch/stack_p4.csv", nodetype = int, delimiter = ",")
for edge in G4.edges():
    G4[edge[0]][edge[1]]['weight'] = 1
G5 = nx.read_edgelist("Stack-exch/stack_p5.csv", nodetype = int, delimiter = ",")
for edge in G5.edges():
    G5[edge[0]][edge[1]]['weight'] = 1
G6 = nx.read_edgelist("Stack-exch/stack_p6.csv", nodetype = int, delimiter = ",")
for edge in G6.edges():
    G6[edge[0]][edge[1]]['weight'] = 1

G15 = nx.read_edgelist("Stack-exch/stack_Stat.csv", nodetype = int, delimiter = ",")
for edge in G15.edges():
    G15[edge[0]][edge[1]]['weight'] = 1

    
    
## All the nodes are in MasterNodes    
MasterNodes = MasterGraph.nodes()


# In[5]:


### Training - Test split  
'''''
first add all the nodes that are in MasterGraph but not in 
G4
'''''
for i in MasterNodes:
    if i not in G6.nodes():
        G6.add_node(i)
        
adj_sparse = nx.to_scipy_sparse_matrix(G6)


# In[7]:


from gae.preprocessing import mask_test_edges
np.random.seed(0) # make sure train-test split is consistent between notebooks

adj_sparse = nx.to_scipy_sparse_matrix(G6)

adj_train, train_edges, train_edges_false, val_edges, val_edges_false,     test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.0, prevent_disconnect = True)


# In[6]:


# Inspect train/test split
print "Total nodes:", adj_sparse.shape[0]
print "Total edges:", int(adj_sparse.nnz/2) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print "Training edges (positive):", len(train_edges)
print "Training edges (negative):", len(train_edges_false)
print "Validation edges (positive):", len(val_edges)
print "Validation edges (negative):", len(val_edges_false)
print "Test edges (positive):", len(test_edges)
print "Test edges (negative):", len(test_edges_false)


# The positive training edges are in the edgelist "train_edges".
# 
# The negative training edges are in the edgelist "train_edges_false".
# 
# The positive test edges are in the edgelist "test_edges".
# 
# The negative test edges are in the edgelist "test_edges_false".

# Step 2: add all the nodes in G1, G2, G3 that are not present

# In[7]:


'''
add all the nodes that are in the MasterGraph but not in 
G1, G2 and G3
'''
for i in MasterNodes:
    if i not in G1.nodes():
        G1.add_node(i)
    if i not in G2.nodes():
        G2.add_node(i)
    if i not in G3.nodes():
        G3.add_node(i)
    if i not in G4.nodes():
        G4.add_node(i)
    if i not in G5.nodes():
        G5.add_node(i)
    if i not in G15.nodes():
        G15.add_node(i)


# In[8]:


print "Edges before removal: "
print "G1:  ", G1.number_of_edges()
print "G2:  ", G2.number_of_edges()
print "G3:  ", G3.number_of_edges()
print "G4:  ", G4.number_of_edges()
print "G5:  ", G5.number_of_edges()
print "G15:  ", G15.number_of_edges()


'''
for every snapshot, delete all the edges that occur in the 
test set, this is important because the training of node2vec
can only be done on the training network and not on edges that
are used for testing
'''
for i in range(0,len(test_edges)):
        if G1.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G1.remove_edge(test_edges[i, 0], test_edges[i, 1])
        if G2.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G2.remove_edge(test_edges[i, 0], test_edges[i, 1])
        if G3.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G3.remove_edge(test_edges[i, 0], test_edges[i, 1])
        if G4.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G4.remove_edge(test_edges[i, 0], test_edges[i, 1])
        if G5.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G5.remove_edge(test_edges[i, 0], test_edges[i, 1])
        if G15.has_edge(test_edges[i, 0], test_edges[i, 1]):
            G15.remove_edge(test_edges[i, 0], test_edges[i, 1])
            
print "Edges after removal: "
print "G1:  ", G1.number_of_edges()
print "G2:  ", G2.number_of_edges()
print "G3:  ", G3.number_of_edges()
print "G4:  ", G4.number_of_edges()
print "G5:  ", G5.number_of_edges()
print "G15:  ", G15.number_of_edges()


# ## 3. Train node2vec (Learn Node Embeddings)

# In[9]:


import node2vec
from gensim.models import Word2Vec


# In[10]:


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


# In[11]:


# Preprocessing, generate walks
G1_n2v = node2vec.Graph(G1, DIRECTED, P, Q) # create node2vec graph instance
G2_n2v = node2vec.Graph(G2, DIRECTED, P, Q)
G3_n2v = node2vec.Graph(G3, DIRECTED, P, Q)
G4_n2v = node2vec.Graph(G4, DIRECTED, P, Q) 
G5_n2v = node2vec.Graph(G5, DIRECTED, P, Q)
G15_n2v = node2vec.Graph(G15, DIRECTED, P, Q)

G1_n2v.preprocess_transition_probs()
G2_n2v.preprocess_transition_probs()
G3_n2v.preprocess_transition_probs()
G4_n2v.preprocess_transition_probs()
G5_n2v.preprocess_transition_probs()
G15_n2v.preprocess_transition_probs()

walksG1 = G1_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksG2 = G2_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksG3 = G3_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksG4 = G4_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksG5 = G5_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksG15 = G15_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)

walksG1 = [map(str, walk) for walk in walksG1]
walksG2 = [map(str, walk) for walk in walksG2]
walksG3 = [map(str, walk) for walk in walksG3]
walksG4 = [map(str, walk) for walk in walksG4]
walksG5 = [map(str, walk) for walk in walksG5]
walksG15 = [map(str, walk) for walk in walksG15]

# Train skip-gram model
modelG1 = Word2Vec(walksG1, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
modelG2 = Word2Vec(walksG2, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
modelG3 = Word2Vec(walksG3, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
modelG4 = Word2Vec(walksG4, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
modelG5 = Word2Vec(walksG5, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)
modelG15 = Word2Vec(walksG15, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)


# Store embeddings mapping
emb_mappingsG1 = modelG1.wv
emb_mappingsG2 = modelG2.wv
emb_mappingsG3 = modelG3.wv
emb_mappingsG4 = modelG4.wv
emb_mappingsG5 = modelG5.wv
emb_mappingsG15 = modelG15.wv


# In[41]:


adj_sparse.shape[0]


# ## 4. Create Edge Embeddings

# In[12]:


# Create node embeddings matrix (rows = nodes, columns = embedding features)
emb_listG1 = []
emb_listG2 = []
emb_listG3 = []
emb_listG4 = []
emb_listG5 = []
emb_listG15 = []

for node_index in MasterGraph.nodes():
    node_str = str(node_index)
    
    node_embG1 = emb_mappingsG1[node_str]
    node_embG2 = emb_mappingsG2[node_str]
    node_embG3 = emb_mappingsG3[node_str]
    node_embG4 = emb_mappingsG4[node_str]
    node_embG5 = emb_mappingsG5[node_str]
    node_embG15 = emb_mappingsG15[node_str]
    
    emb_listG1.append(node_embG1)
    emb_listG2.append(node_embG2)
    emb_listG3.append(node_embG3)
    emb_listG4.append(node_embG4)
    emb_listG5.append(node_embG5)
    emb_listG15.append(node_embG15)
    
emb_matrixG1 = np.vstack(emb_listG1)
emb_matrixG2 = np.vstack(emb_listG2)
emb_matrixG3 = np.vstack(emb_listG3)
emb_matrixG4 = np.vstack(emb_listG4)
emb_matrixG5 = np.vstack(emb_listG5)
emb_matrixG15 = np.vstack(emb_listG15)


# In[13]:


# Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings_dynamic(edge_list):
    embs = []
    for edge in edge_list:
        
        node1 = edge[0]
        node2 = edge[1]
        
        embG1_1 = emb_matrixG1[node1]
        embG1_2 = emb_matrixG1[node2]
        
        embG2_1 = emb_matrixG2[node1]
        embG2_2 = emb_matrixG2[node2]
        
        embG3_1 = emb_matrixG3[node1]
        embG3_2 = emb_matrixG3[node2]
        
        embG4_1 = emb_matrixG4[node1]
        embG4_2 = emb_matrixG4[node2]
        
        embG5_1 = emb_matrixG5[node1]
        embG5_2 = emb_matrixG5[node2]
        
        edge_embG1 = np.multiply(embG1_1, embG1_2)
        edge_embG2 = np.multiply(embG2_1, embG2_2)
        edge_embG3 = np.multiply(embG3_1, embG3_2)
        edge_embG4 = np.multiply(embG4_1, embG4_2)
        edge_embG5 = np.multiply(embG5_1, embG5_2)
        
        edge_emb = np.hstack((edge_embG1,edge_embG2, edge_embG3, edge_embG4, edge_embG5))
        embs.append(edge_emb)
        
    embs = np.array(embs)
    
    return embs


# In[14]:


def get_edge_embeddings_static(edge_list):
    embs_s = []
    for edge in edge_list:
        
        node1 = edge[0]
        node2 = edge[1]
        
        embG15_1 = emb_matrixG15[node1]
        embG15_2 = emb_matrixG15[node2]
        
        edge_embG15 = np.multiply(embG15_1, embG15_2)
        embs_s.append(edge_embG15)
        
    embs_s = np.array(embs_s)
    
    return embs_s


# In[15]:


## DYNAMIC
# Train-set edge embeddings
pos_train_edge_embs_d = get_edge_embeddings_dynamic(train_edges)
neg_train_edge_embs_d = get_edge_embeddings_dynamic(train_edges_false)
train_edge_embs_d = np.concatenate([pos_train_edge_embs_d, neg_train_edge_embs_d])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

# Val-set edge embeddings, labels
pos_val_edge_embs_d = get_edge_embeddings_dynamic(val_edges)
neg_val_edge_embs_d = get_edge_embeddings_dynamic(val_edges_false)
val_edge_embs_d = np.concatenate([pos_val_edge_embs_d, neg_val_edge_embs_d])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

# Test-set edge embeddings, labels
pos_test_edge_embs_d = get_edge_embeddings_dynamic(test_edges)
neg_test_edge_embs_d = get_edge_embeddings_dynamic(test_edges_false)
test_edge_embs_d = np.concatenate([pos_test_edge_embs_d, neg_test_edge_embs_d])

# Create val-set edge labels: 1 = real edge, 0 = false edge
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])


# In[16]:


## STATIC
# Train-set edge embeddings
pos_train_edge_embs_s = get_edge_embeddings_static(train_edges)
neg_train_edge_embs_s = get_edge_embeddings_static(train_edges_false)
train_edge_embs_s = np.concatenate([pos_train_edge_embs_s, neg_train_edge_embs_s])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

# Val-set edge embeddings, labels
pos_val_edge_embs_s = get_edge_embeddings_static(val_edges)
neg_val_edge_embs_s = get_edge_embeddings_static(val_edges_false)
val_edge_embs_s = np.concatenate([pos_val_edge_embs_s, neg_val_edge_embs_s])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

# Test-set edge embeddings, labels
pos_test_edge_embs_s = get_edge_embeddings_static(test_edges)
neg_test_edge_embs_s = get_edge_embeddings_static(test_edges_false)
test_edge_embs_s = np.concatenate([pos_test_edge_embs_s, neg_test_edge_embs_s])

# Create val-set edge labels: 1 = real edge, 0 = false edge
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])


# ## 5. Evaluate Edge Embeddings

# First, the basic topological classifiers are calculated for the test and training set.

# In[17]:


# Train logistic regression classifier on train-set edge embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


edge_classifier_lr_stat = LogisticRegression(random_state=0)
edge_classifier_lr_stat.fit(train_edge_embs_s, train_edge_labels)

edge_classifier_lr_dyn = LogisticRegression(random_state=0)
edge_classifier_lr_dyn.fit(train_edge_embs_d, train_edge_labels)

edge_classifier_RF_stat = RandomForestClassifier(n_estimators = 50)
edge_classifier_RF_stat.fit(train_edge_embs_s, train_edge_labels)

edge_classifier_RF_dyn = RandomForestClassifier(n_estimators = 50)
edge_classifier_RF_dyn.fit(train_edge_embs_d, train_edge_labels)


# In[18]:


##  Gradient Boosted Regression Trees
from sklearn.ensemble import GradientBoostingClassifier
edge_classifier_gb_stat = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=8, random_state=0).fit(train_edge_embs_s, train_edge_labels)
edge_classifier_gb_dyn = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=8, random_state=0).fit(train_edge_embs_d, train_edge_labels)


# In[72]:


# Predicted edge scores: probability of being of class "1" (real edge)
# val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
# val_roc = roc_auc_score(val_edge_labels, val_preds)
# val_ap = average_precision_score(val_edge_labels, val_preds)


# In[19]:


# Predicted edge scores: probability of being of class "1" (real edge)
test_preds_lr_s = edge_classifier_lr_stat.predict_proba(test_edge_embs_s)[:, 1]
test_roc_lr_s = roc_auc_score(test_edge_labels, test_preds_lr_s)
test_ap_lr_s = average_precision_score(test_edge_labels, test_preds_lr_s)

test_preds_lr_d = edge_classifier_lr_dyn.predict_proba(test_edge_embs_d)[:, 1]
test_roc_lr_d = roc_auc_score(test_edge_labels, test_preds_lr_d)
test_ap_lr_d = average_precision_score(test_edge_labels, test_preds_lr_d)

test_preds_rf_s = edge_classifier_RF_stat.predict_proba(test_edge_embs_s)[:, 1]
test_roc_rf_s = roc_auc_score(test_edge_labels, test_preds_rf_s)
test_ap_rf_s = average_precision_score(test_edge_labels, test_preds_rf_s)

test_preds_rf_d = edge_classifier_RF_dyn.predict_proba(test_edge_embs_d)[:, 1]
test_roc_rf_d = roc_auc_score(test_edge_labels, test_preds_rf_d)
test_ap_rf_d = average_precision_score(test_edge_labels, test_preds_rf_d)

test_preds_gb_s = edge_classifier_gb_stat.predict_proba(test_edge_embs_s)[:, 1]
test_roc_gb_s = roc_auc_score(test_edge_labels, test_preds_gb_s)
test_ap_gb_s = average_precision_score(test_edge_labels, test_preds_gb_s)

test_preds_gb_d = edge_classifier_gb_dyn.predict_proba(test_edge_embs_d)[:, 1]
test_roc_gb_d = roc_auc_score(test_edge_labels, test_preds_gb_d)
test_ap_gb_d = average_precision_score(test_edge_labels, test_preds_gb_d)


# In[20]:


# print 'node2vec Validation ROC score: ', str(val_roc)
# print 'node2vec Validation AP score: ', str(val_ap)
print 'node2vec Test ROC score logistic regression static: ', str(test_roc_lr_s)
print 'node2vec Test ROC score logistic regression dynamic: ', str(test_roc_lr_d)
print 'node2vec Test ROC score random forest static: ', str(test_roc_rf_s)
print 'node2vec Test ROC score random forest dynamic: ', str(test_roc_rf_d)
print 'node2vec Test ROC score gradient boosting static: ', str(test_roc_gb_s)
print 'node2vec Test ROC score gradient boosting dynamic: ', str(test_roc_gb_d)
print 'node2vec Test AP score logistic regression static: ', str(test_ap_lr_s)
print 'node2vec Test AP score logistic regression dynamic: ', str(test_ap_lr_d)
print 'node2vec Test AP score random forest static: ', str(test_ap_rf_s)
print 'node2vec Test AP score random forest dynamic: ', str(test_ap_rf_d)
print 'node2vec Test AP score gradient boosting static: ', str(test_ap_gb_s)
print 'node2vec Test AP score gradient boosting dynamic: ', str(test_ap_gb_d)


# In[21]:


## ROC curve

fpr_lr_s, tpr_lr_s, _ = roc_curve(test_edge_labels, test_preds_lr_s)
fpr_lr_d, tpr_lr_d, _ = roc_curve(test_edge_labels, test_preds_lr_d)
fpr_rf_s, tpr_rf_s, _ = roc_curve(test_edge_labels, test_preds_rf_s)
fpr_rf_d, tpr_rf_d, _ = roc_curve(test_edge_labels, test_preds_rf_d)
fpr_gb_s, tpr_gb_s, _ = roc_curve(test_edge_labels, test_preds_gb_s)
fpr_gb_d, tpr_gb_d, _ = roc_curve(test_edge_labels, test_preds_gb_d)


fig_roc = plt.figure()

plt.plot([0,1], [0, 1], 'k--')
plt.step(fpr_lr_s, tpr_lr_s, color = "b", alpha = 1, where = 'post', label = "Logistic Regression/Static")
plt.step(fpr_lr_d, tpr_lr_d, color = "lime", alpha = 1, where = 'post', label = "Logistic Regression/Dynamic")
plt.step(fpr_rf_s, tpr_rf_s, color = "salmon", alpha = 1, where = 'post', label = "Random Forest/Static")
plt.step(fpr_rf_d, tpr_rf_d, color = "olive", alpha = 1, where = 'post', label = "Random Forest/Dynamic")
plt.step(fpr_gb_s, tpr_gb_s, color = "red", alpha = 1, where = 'post', label = "Gradient Boosting/Static")
plt.step(fpr_gb_d, tpr_gb_d, color = "grey", alpha = 1, where = 'post', label = "Gradient Boosting/Dynamic")



plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('ROC curve Stackexchange')

plt.legend()

fig_roc.savefig("ROC_stackexchange.png")



# In[22]:


## Precision - Recall curve

precision_lr_s, recall_lr_s, _ = precision_recall_curve(test_edge_labels, test_preds_lr_s)
precision_rf_s, recall_rf_s, _ = precision_recall_curve(test_edge_labels, test_preds_rf_s)
precision_lr_d, recall_lr_d, _ = precision_recall_curve(test_edge_labels, test_preds_lr_d)
precision_rf_d, recall_rf_d, _ = precision_recall_curve(test_edge_labels, test_preds_rf_d)
precision_gb_s, recall_gb_s, _ = precision_recall_curve(test_edge_labels, test_preds_gb_s)
precision_gb_d, recall_gb_d, _ = precision_recall_curve(test_edge_labels, test_preds_gb_d)

fig_aupr = plt.figure()

plt.step(recall_lr_s, precision_lr_s, color="b", alpha=1, where='post', label = "Logistic Regression/Static")
plt.step(recall_lr_d, precision_lr_d, color="lime", alpha=1, where='post', label = "Logistic Regression/Dynamic")

plt.step(recall_rf_s, precision_rf_s, color="salmon", alpha=1, where='post', label = "Random Forest/Static")
plt.step(recall_rf_d, precision_rf_d, color="olive", alpha=1, where='post', label = "Random Forest/Dynamic")

plt.step(recall_gb_s,precision_gb_s,  color="red", alpha=1, where='post', label = "Gradient Boosting/Static")
plt.step( recall_gb_d,precision_gb_d, color="grey", alpha=1, where='post', label = "Gradient Boosting/Dynamic")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve Stackexchange')

plt.legend()

fig_aupr.savefig("AUPR_stackexchange.png")

