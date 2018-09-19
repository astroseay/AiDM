import numpy as np
import pandas as pd
import time
from sklearn import linear_model
#RW gets 1.01, 1.01
def mean_user(x):
    user_count=np.bincount(x[0])
    zeros_train = np.array(np.where(user_count[1:len(user_count)] == 0))
    non_zero_train = np.array([np.where(user_count[1:len(user_count)] != 0)])
    times_user_train_correct = np.delete(user_count[1:len(user_count)], zeros_train)
    mean_user = np.array(x.groupby([0])[2].mean())
    full = np.repeat(mean_user,times_user_train_correct)
    return np.array(full)

def mean_movie(x):
    movie_count = np.bincount(x[1])
    zeros_train = np.array(np.where(movie_count[1:len(movie_count)] == 0))
    non_zero_train = np.array([np.where(movie_count[1:len(movie_count)] != 0)])
    times_movie_train_correct = np.delete(movie_count[1:len(movie_count)], zeros_train)
    mean_movie = np.array(x.groupby([1])[2].mean())
    full = np.repeat(mean_movie, times_movie_train_correct)
    return np.array(full)

def combo(fn):
    df = pd.DataFrame(fn)
    ratings_user=pd.DataFrame(fn)
    ratings_user=ratings_user.append(ratings_user)
    user_average = df.groupby(by=0, as_index=False)[2].mean()
    user_average = user_average.append(user_average)
    ratings_movie=pd.DataFrame(fn)
    ratings_movie=ratings_movie.append(ratings_movie)
    movie_average = df.groupby(by=1, as_index=False)[2].mean()
    movie_average = movie_average.append(movie_average)
    global_average = np.mean(fn[:,2])

    nfolds = 5

    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)
    alpha=np.zeros(nfolds)
    beta=np.zeros(nfolds)
    gamma=np.zeros(nfolds)

    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(fn))]
    np.random.shuffle(seqs)

    start_time = time.time()
    print ('Recommendations from a combination of user and movie averages:')
    for fold in range(nfolds):

        train_set=np.array([x!=fold for x in seqs])
        test_set=np.array([x==fold for x in seqs])
        train = pd.DataFrame(ratings_movie.iloc[test_set], columns=[0, 1, 2], dtype=int)
        test = pd.DataFrame(ratings_movie.iloc[test_set], columns=[0, 1, 2], dtype=int)
        X = np.vstack([np.array(mean_user(train)), np.array(mean_movie(train))]).T
        reg = linear_model.LinearRegression()

        reg.fit(X[:,:],np.array(train[2]))

        alpha[fold] = reg.coef_[0]  # coeff of alpha
        beta[fold] = reg.coef_[1]  # coeff of beta
        gamma[fold] = reg.intercept_  # coeff of the intercept (gamma)
        #print alpha[fold], beta, gamma
        # applying the values above to the formula in the book

        pred_train = alpha[fold] * mean_user((train)) + beta[fold] * mean_movie((train)) + gamma[fold]
        pred_test= alpha[fold] * mean_user((test)) + beta[fold] * mean_movie((test)) + gamma[fold]
        pred_train[pred_train > 5] = 5
        pred_train[pred_train < 1] = 1
        pred_test[pred_test>5]=5
        pred_test[pred_test<1]=1

        err_train[fold] = np.sqrt(np.mean((np.array(train[2])-pred_train)**2))
        err_test[fold] = np.sqrt(np.mean((np.array(test[2])-pred_test)**2))
        mae_train[fold] = np.mean(np.abs(np.array(train[2])-pred_train))
        mae_test[fold] = np.mean(np.abs(test[2]-pred_test))
        print("Fold " + str(fold+1) + ": RMSE_train = " + str(err_train[fold]) + "; RMSE_test = " + str(err_test[fold]))



    print("\n")
    print('Mean error on TRAIN: '+ str(np.mean(err_train)))
    print('Mean error on  TEST: ' + str(np.mean(err_test)))
    print ('MAE on TRAIN: ' + str(np.mean(mae_train)))
    print ('MAE on  TEST: ' + str(np.mean(mae_test)))
    print ("alpha =", np.mean(alpha), "; beta =",np.mean(beta) , "; gamma =", np.mean(gamma))

    print("Linear regression runtime:  %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print('MATRIX FACTORIZATION. Params:num_factors=10,num_iter=75, regularization=0.05, learn_rate=0.005,np.random.seed(17)')

def matrixFactorization(train, test, num_factors=10, num_iter=75, regularization=0.05, learn_rate=0.005):
    np.random.seed(1)
    train=np.array(train)
    test=np.array(test)
    U=np.random.rand(max(np.max(train[:,0]), np.max(test[:,0]) + 1), num_factors)
    M=np.random.rand(num_factors, max(np.max(train[:,1]), np.max(test[:,1])) + 1)
    for i in range(num_iter):

        for j in range(len(train)):
            eTimes2 = 2 * (train[j,2] - np.dot(U[train[j,0],:], M[:,train[j,1]]))
            # compute the gradient of e**2 to M before changing U (negative of the gradient)
            mGradient = eTimes2 * U[train[j,0],:]
            uGradient = eTimes2 * M[:,train[j,1]]
            U[train[j, 0], :] += learn_rate * (uGradient - regularization * U[train[j, 0], :])
            M[:, train[j, 1]] += learn_rate * (mGradient - regularization * M[:,train[j, 1]])

        # calculate estimated ratings

        ER = np.dot(U, M)

        # make prediction for train
        predictionTrain = np.zeros(len(train))
        for i in range(len(train)):
            predictionTrain[i] = ER[train[i, 0], train[i, 1]]
            if predictionTrain[i] > 5:
                predictionTrain[i] = 5
            if predictionTrain[i] < 1:
                predictionTrain[i] = 1

        # make prediction for test
        prediction = np.zeros(len(test))
        for i in range(len(test)):
            prediction[i] = ER[test[i,0], test[i, 1]]
            if prediction[i] > 5:
                prediction[i] = 5
            if prediction[i] < 1:
                prediction[i] = 1

        return (predictionTrain, prediction)

        # matrixFactorizationGravity = matrixFactorization_gravity(train, test)

for fold in range(nfolds):
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train = pd.DataFrame(ratings_user.iloc[test_sel], columns=[0, 1, 2], dtype=int)
    test = pd.DataFrame(ratings_user.iloc[test_sel], columns=[0, 1, 2], dtype=int)
    #matrixFactorization_gravity(train, test, num_factors=1, num_iter=75, regularization=0.05, learn_rate=0.005)
    err_train[fold] = np.sqrt(np.mean((np.array(train[2]) - matrixFactorization(train, test, num_factors=10, num_iter=75, regularization=0.05, learn_rate=0.005)[0]) ** 2))
    err_test[fold]= np.sqrt(np.mean((np.array(test[2])-matrixFactorization(train, test, num_factors=10, num_iter=75, regularization=0.05, learn_rate=0.005)[1])**2))
    mae_train[fold] = np.mean(np.abs(np.array(train[2])-matrixFactorization(train, test, num_factors=10, num_iter=75, regularization=0.05, learn_rate=0.005)[0]))
    mae_test[fold] = np.mean(np.abs(np.array(test[2])-matrixFactorization(train, test, num_factors=10, num_iter=75, regularization=0.05, learn_rate=0.005)[1]))
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))



print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))
print ('MAE on TRAIN:' + str(np.mean(mae_train)))
print ('MAE on TEST:' + str(np.mean(mae_train)))
print("Matrix Factorization:  %s seconds ---" % (time.time() - start_time))
