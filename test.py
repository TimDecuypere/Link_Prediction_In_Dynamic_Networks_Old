
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


print 'all good'