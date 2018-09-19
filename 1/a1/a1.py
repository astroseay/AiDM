"""
Created 15/09/2018
@author: anon

Assignment 1: Recommender system

Citations: Wolacyzk demo_avg_rating.py and demo_least_squares.py
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import global_avg
import movie_avg
import user_avg
import combo
# import matrix_fact
# import als

# np.set_printoptions(np.inf)
# f_path = '/Users/seayc/Documents/school/masters/classes/year_one/aidm/hw/1/data/'
# ratings = np.genfromtxt(f_path+'ratings.dat',delimiter='::',dtype='int')
# np.save('rate',ratings)

def recommender():
    #split data into 5 train and test folds
    nfolds=5

    #allocate memory for results:
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)

    #to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    #for each fold:
    for fold in range(nfolds):
        train_sel=np.array([x!=fold for x in seqs])
        test_sel=np.array([x==fold for x in seqs])
        train=ratings[train_sel]
        test=ratings[test_sel]

    #calculate model parameters: mean rating over the training set:
        gmr=np.mean(train[:,2])

    #apply the model to the train set:
        err_train[fold]=np.sqrt(np.mean((train[:,2]-gmr)**2))

    #apply the model to the test set:
        err_test[fold]=np.sqrt(np.mean((test[:,2]-gmr)**2))

    #print errors:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

    #print the final conclusion:
    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))

ratings = np.load('rate.npy')
global_avg.global_avg(ratings)
user_avg.user_avg(ratings)
movie_avg.movie_avg(ratings)
combo.combo(ratings)
#matrix_fact
#als
# print(ratings[:,2])
