import numpy as np
import pandas as pd
import time

def mean_movie(x):
    movie_count = np.bincount(x[1])
    zeros_train = np.array(np.where(movie_count[1:len(movie_count)] == 0))
    non_zero_train = np.array([np.where(movie_count[1:len(movie_count)] != 0)])
    times_movie_train_correct = np.delete(movie_count[1:len(movie_count)], zeros_train)
    mean_movie = np.array(x.groupby([1])[2].mean())
    full = np.repeat(mean_movie, times_movie_train_correct)
    return np.array(full)

def movie_avg(fn):
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)

    start_time = time.time()
    print ('Movie Average as a Recommender:')
    for fold in range(nfolds):
        train_set=np.array([x!=fold for x in seqs])
        test_set=np.array([x==fold for x in seqs])
        train_movies=pd.DataFrame(ratings_movie.iloc[train_set],columns=[0, 1, 2],dtype=int)
        test_movies=pd.DataFrame(ratings_movie.iloc[test_set],columns=[0, 1, 2],dtype=int)
        #apply to train
        err_train[fold] = np.sqrt(np.mean((np.array(train_movies[2]) - mean_movie(train_movies)) ** 2))

        # apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((np.array(test_movies[2]) - mean_movie(test_movies)) ** 2))

        mae_train[fold] = np.mean(np.abs(np.array(train_movies[2]) - mean_movie(train_movies)))
        mae_test[fold] =  np.mean(np.abs(np.array(test_movies[2]) - mean_movie(test_movies)))

        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))



    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print ('MAE on TRAIN: ' + str(np.mean(mae_train)))
    print ('MAE on TEST: ' + str(np.mean(mae_test)))

    print("Movie avg:  %s seconds ---" % (time.time() - start_time))
