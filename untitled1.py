# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:18:29 2018

@author: Sam
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pickle

Gr = nx.read_edgelist("may_data.csv", nodetype=int, create_using=nx.DiGraph())

print Gr.number_of_edges()
print Gr.number_of_nodes()