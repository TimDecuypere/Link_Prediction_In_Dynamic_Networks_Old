# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:02:31 2018

@author: Tim
"""
import numpy as np
from numpy import genfromtxt
my_data = genfromtxt('soc-wiki-elec.edges', delimiter=' ')

my_data=my_data.astype(int)

print my_data

np.savetxt("m_wiki_elec.csv", my_data[1:,0:2] , delimiter=",",fmt='%1.0f')
my_data_split = np.split(my_data,4)
np.savetxt("m13_wiki_elec.csv", np.vstack((my_data_split[0],my_data_split[1],my_data_split[2]))[:,0:2] , delimiter=",",fmt="%1.0f")

np.savetxt("m1_wiki_elec.csv", my_data_split[0][:,0:2], delimiter=",",fmt='%1.0f')
np.savetxt("m2_wiki_elec.csv", my_data_split[1][:,0:2], delimiter=",",fmt='%1.0f')
np.savetxt("m3_wiki_elec.csv", my_data_split[2][:,0:2], delimiter=",",fmt='%1.0f')
np.savetxt("m4_wiki_elec.csv", my_data_split[3][:,0:2], delimiter=",",fmt='%1.0f')