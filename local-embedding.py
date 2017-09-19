#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
import os
import gensim
import logging
import codecs
import time
from collections import defaultdict
from math import*
 

 


word2vec = gensim.models.Word2Vec

def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, str):
        return text
    return text.decode("utf-8", "replace")

gensim.utils.to_unicode = any2unicode

# path to the already constructed Models
MODELS_DIR = 'models/glove/'


def get_embeddings(model_name, model, positive_label_lists, negative_label_lists=[], topn=20):
    row_collist_map = defaultdict(list)

    for col,label_list in enumerate(positive_label_lists):
        print("\n>>", label_list)
        line = []
        row_collist_map[0].extend([model_name, '', ''])
        row_collist_map[1].extend(['', '', ''])
        try:
            for row, sim in enumerate(model.most_similar(positive=label_list, negative=negative_label_lists, topn=topn)):
                res = sim[0].rjust(30) + "\t%.3f" % sim[1]
                print(res)
                row_collist_map[row+2].extend([str(sim[0]),"%.3f" % sim[1],''])

        except KeyError:
            print ('>>> ** KeyError ** -- word probably not in vocabulary')

        print("Wrote", out_file)

    return row_collist_map


if __name__ == '__main__':
    # label_list_map = {
    #       'test': [['biscuit']]}

    print("\n=================")
    # print("Models:")

    model_files = ['processed_food.ALL.concepts.500d.bin'] # ', glove/glove.6B.300d.txt.gz'] processed_food.ALL.concepts.500d

    for file in model_files:
        print("\n Model file:", '.'.join(file.split('.')[1:]))
        model_name = file
        start = time.time()

        base_file = file.split("/")[-1].rstrip('.txt.gz')

        model = word2vec.load_word2vec_format(file, binary=True)

        model.init_sims(replace=True)

        print("\n=================")
        

########################################################################################################

#4607
###############################################
vector={}
fixed='sugar'
Input=[['diabetic',1.0],['diabetes',1.0],['hypertension',0.8] ,['insulin',0.8],['hypoglycemia',0.8],['obesity',0.8],['carbohydrate',0.69],['illness',0.6], ['blood_sugar',0.6], ['syrup',0.6] ,['cholesterol',0.5], ['diet',0.5], ['kidney',0.4], ['nutrition',0.4]]
firstVec=[]
NewVectors=[]
for element in Input:
    a=((numpy.array(model[element[0]])-numpy.array(model[fixed]))*element[1]) + numpy.array(model[fixed])
    vector[element[0]]=a/(numpy.linalg.norm(a))
    NewVectors.append(vector[element[0]])
    firstVec.append(model[element[0]])
NewVectors.append(model[fixed])
firstVec.append(model[fixed])
Compare=[]
for sim in model.most_similar(positive=[fixed], negative=[], topn=4607):
    Compare.append(model[sim[0]])   
Compare.append(model[fixed])


target=(numpy.array(model[fixed]))
for key in vector.keys():
   target=target+numpy.array(vector[key])
target=target / (numpy.linalg.norm(target))

######################################################################################
A= (numpy.array(NewVectors)).T
P=numpy.dot(NewVectors,A)
v=(numpy.array([target])).T
q_tran=(numpy.dot(target,A))*(-1.0)
q=(numpy.array([q_tran])).T


P=numpy.array(P)
P = P.astype(numpy.double)
P = matrix(P)


q=numpy.array(q)
q = q.astype(numpy.double)
q = matrix(q)

G=[[0.0] for i in range(len(Input)+1)]
G=matrix(G)

h=matrix([1.0])

b=matrix([1.0])

A=[[1.0] for i in range(len(Input)+1)]
A=matrix(A)

sol = solvers.qp(P,q,G,h,A,b)
print sol['x']
firstVec=numpy.array(firstVec)
firstVec=firstVec.T

NewPoint=numpy.dot(firstVec,sol['x'])
NewPoint=NewPoint/(numpy.linalg.norm(NewPoint))
NewPoint=(numpy.array(NewPoint)).T

NewPoint= numpy.array(NewPoint)
target=NewPoint

########################################################################################
semi_final=[]
for element in Compare:
    element=numpy.array(element)
    dist=numpy.linalg.norm(target - element)
    semi_final.append(dist)
Final={}  
i=0
for sim in model.most_similar(positive=[fixed], negative=[], topn=4607):
    Final[sim[0]]=semi_final[i]
    i=i+1

Final[Input[0][0]]=semi_final[4607]
K= sorted(Final.values())
#K=K[0:500]
for m in K:
#    for key in Final.keys():

        key = [x for x in Final if Final[x] == m][0]

        weight = numpy.inner(target,model[key])

        if weight < 0.25:
            break

#        if Final[key]== m: 

        print key.rjust(30),",\t %.3f" % weight #numpy.inner(target,model[key])