from NeuralNetwork import ANN as model
import numpy as np

EPSILON = 1e-10

# def cross_validation(X, y, splits = [0.7, 0.15, 0.15], k=5, hyperparameters = [0.1, 0.001, "mse", "mse_deriv"]):
#     """
#     Performs cross-validation on the given data.
    
#     Parameters:
#         X (ndarray): Input data array of shape (n_samples, n_features).
#         y (ndarray): Target array of shape (n_samples, ).
#         splits (ndarray): split into train, valid and test set respectively
#         k (int): Number of folds for cross-validation.
#         grids (ndarray): arrays of hyperparameters
    
#     Returns:
#         scores (int): the accuracy score for the best hyperparameter
#     """    
#     assert(np.sum(splits), 1)
#     # Shuffle the data randomly
#     idx = np.random.permutation(X.shape[0])
#     X, y = X[idx], y[idx]

#     num_sample = np.shape(X)[0]
#     num_sample_train = int(num_sample * splits[0])
#     num_sample_validation = int(num_sample * splits[1])
#     num_sample_test = num_sample - num_sample_train - num_sample_validation
    
#     # Reserve for final test
#     X_test_set = X[:num_sample_test]
#     y_test_set = y[:num_sample_test]

#     # starts from the index num_sample_test (since the first num_sample_test has been reserved for testing)
#     score, hyper_parameters = k_fold_cross_validation(X[num_sample_test:], y[num_sample_test:], model, k, hyperparameters)

#     #Train on the train and validation set with BEST hyper parameters
#     model.train(X[num_sample_test:], y[num_sample_test:], hyper_parameters[0], hyper_parameters[1], hyper_parameters[2], "mse_deriv", hyper_parameters[3])

#     results = model.predict(X_test_set)
#     score = np.count_nonzero(results - y_test_set < EPSILON) / np.shape(X_test_set)[0]

#     return score

def k_fold_cross_validation(X, y, model, loss_func, loss_deriv, k=5, hyper_parameters = [0.5, 0.001, 3]):
    """
    Performs k-fold cross-validation on the given data.
    
    Parameters:
        X (ndarray): Input data array of shape (n_samples, n_features).
        y (ndarray): Target array of shape (n_samples, ).
        k (int): Number of folds for cross-validation.
    
    Returns:
        score (float): BEST accuracy score for in all grid
        hyper_parameter (1darray): BEST set of hyper parameters found in the grid space
    """
    # Shuffle the data randomly
    idx = np.random.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    
    # Split the data into k folds
    fold_size = X.shape[0] // k
    X_folds = [X[i*fold_size:(i+1)*fold_size] for i in range(k)]
    y_folds = [y[i*fold_size:(i+1)*fold_size] for i in range(k)]
    
    scores = []
    prev_score = -1
    score = 0

    # loops through all the folds
    for i in range(k):
        # Use the i-th fold as the validation set, and the rest as the training set
        X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(k) if j != i])
        X_val = X_folds[i]
        y_val = y_folds[i]
        
        # Train the model on the training set
        total_loss = model.train(X_train, 
                    y_train, 
                    X_val[:1],
                    y_val[:1],
                    # Learning rate
                    hyper_parameters[0], 
                    # Threshhold
                    hyper_parameters[1], 
                    # Loss function
                    loss_func,
                    # Loss function deriv
                    loss_deriv,
                    hyper_parameters[2]
                    )
        
        # Evaluate the model on the validation set and store the score
        results = model.predictBatch(X_val)
        for i in range(len(results)):
            j=np.argmax(results[i])
            results[i,:] = 0
            results[i,j] = 1
             
        # Using accuracy
        count=0
        for i in range(len(results)):
            count += np.dot(results[i], y_val[i])
        scores.append(count/len(results))
    
    score = np.mean(scores)

    # if score > prev_score:
    #   prev_score = score
    #   hyper_parameters = current_set_of_hp

    return score 


#test:
# dataX = open("../data/features.txt", 'r')
# dataY = open("../data/targets.txt", 'r')

# X = []
# y = []

# for id, temp_x in enumerate(dataX):
#     X.append(np.fromstring(temp_x, sep=','))

# for temp_y in dataY:
#     y.append(np.fromstring(temp_y,sep=','))
# X = np.array(X)
# y = np.array(y)

# k_fold_cross_validation(X, y)
