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
from scipy.optimize import minimize 
 

 


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

    model_files = ['processed_food.ALL.concepts.500d.bin'] # ', glove/glove.42B.300d.txt.gz'] processed_food.ALL.concepts.500d

    for file in model_files:
        print("\n Model file:", '.'.join(file.split('.')[1:]))
        model_name = file
        start = time.time()

        base_file = file.split("/")[-1].rstrip('.txt.gz')

        model = word2vec.load_word2vec_format(file, binary=True)

        model.init_sims(replace=True)

        print("\n=================")


########################################################################################################
#4607   number of words in processed_food.ALL.concepts.500d model
#215266 number of words in glove/glove.6B.300d model
###############################################

#Input=[['diabetes',0.696], ['diabetic',0.694], ['blood_sugar',0.678], ['insulin_resistance',0.668], ['hypoglycemia',0.667], ['insulin',0.643], ['blood_sugar_level',0.641], ['metabolic_syndrome',0.608], ['processed_food_disease',0.608], ['chronic_metabolic_disease',0.598], ['hypertension',0.598], ['sugar_level',0.591], ['diabesity',0.578], ['risk_for_heart_disease',0.574], ['sharma_obesity',0.565]]
Input=[['sugar',1.0],['diabetic',1.0],['diabetes',1.0],['hypertension',0.8] ,['insulin',0.8],['hypoglycemia',0.8],['obesity',0.8],['carbohydrate',0.69],['illness',0.6], ['blood_sugar',0.6], ['syrup',0.6] ,['cholesterol',0.5], ['diet',0.5], ['kidney',0.4], ['nutrition',0.4]]
r1=1
r2=1
Q=[]
for element in Input:
    a=model[element[0]]*sqrt((element[1])**r1)
    Q.append(a) 
Q=numpy.matrix(Q)  
Q=Q.T
Q= -Q
I=[1 for i in range(len(Input))] 
I=numpy.matrix(I)

fun = lambda x: sum((x*Q[:,i])**r2 for i in range(len(Input)))
cons = ({'type': 'eq', 'fun': lambda x: sum(x[i]**2 for i in range(len(model[Input[0][0]])))-1}) #(x[0]**2) + (x[1]**2) + (x[2]**2)+ (x[3]**2) - 1})

init=[]
init.append(1)
for i in range(1,len(model[Input[0][0]])):
    init.append(0)
res = minimize(fun, init, method='SLSQP',constraints=cons)

target = res.x

   
Compare=[]
for sim in model.most_similar(positive=[Input[0][0]], negative=[], topn=4607):
    Compare.append(model[sim[0]])   
Compare.append(model[Input[0][0]])


semi_final=[]
for element in Compare:
    element=numpy.array(element)
    dist=numpy.linalg.norm(target - element)
    semi_final.append(dist)
Final={}  
i=0
for sim in model.most_similar(positive=[Input[0][0]], negative=[], topn=4607):
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















