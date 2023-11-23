import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score

# Tambahkan perhitungan korelasi Pearson antara fitur-fitur
def calculate_pearson_correlation(xtrain):
    num_features = xtrain.shape[1]
    pearson_correlation = np.zeros((num_features, num_features))
    
    for i in range(num_features):
        for j in range(num_features):
            pearson_correlation[i, j] = np.corrcoef(xtrain[:, i], xtrain[:, j])[0, 1]
    
    return pearson_correlation

def optimized_combined_kernel(X, Y):
    # Kernel Linier
    K_lin = np.dot(X, Y.T)
    
    # Kernel RBF dengan gamma yang dioptimalkan
    gamma = 0.3
    squared_distance = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Y**2, 1) - 2 * np.dot(X, Y.T)
    K_rbf = np.exp(-gamma * squared_distance)
    
    # Gabungan kernel linier dan RBF
    beta = 0.001
    return beta * K_lin + (1 - beta) * K_rbf

# error rate
# def error_rate(xtrain, ytrain, x, opts):
#     k = opts['k']
#     fold = opts['fold']
#     xtrain = fold['x']
#     ytrain = fold['y']
#     num = np.size(xtrain, 0)
#     xtrain = xtrain[:, x == 1]
#     ytrain = ytrain.reshape(num)

#     ss = StratifiedKFold(n_splits=10, shuffle=True)  # Menggunakan StratifiedKFold
#     clf = SVC(kernel='linear', random_state=42)
#     correct = 0

#     for train, test in ss.split(xtrain, ytrain):  # Memberikan ytrain ke split
#         clf.fit(xtrain[train], ytrain[train])
#         y_predict = clf.predict(xtrain[test])
#         acc = accuracy_score(ytrain[test], y_predict)
#         correct = correct + acc

#     error = 1 - float(correct / 10)
#     return error

def error_rate(xtrain, ytrain, x, opts):
    k = opts['k']
    fold = opts['fold']
    xtrain = fold['x']
    ytrain = fold['y']
    num = np.size(xtrain, 0)
    xtrain = xtrain[:, x == 1]
    ytrain = ytrain.reshape(num)

    ss = StratifiedKFold(n_splits=10, shuffle=True)  # Menggunakan StratifiedKFold
    clf = SVC(kernel=optimized_combined_kernel, random_state=42)
    f1_scores = []

    for train, test in ss.split(xtrain, ytrain):  # Memberikan ytrain ke split
        clf.fit(xtrain[train], ytrain[train])
        y_predict = clf.predict(xtrain[test])
        f1 = f1_score(ytrain[test], y_predict, average='weighted')  # Menggunakan F1-score sebagai metrik
        f1_scores.append(f1)

    error = 1 - np.mean(f1_scores)
    return error

def Fun(xtrain, ytrain, x, opts):
    alpha = 0.99
    beta = 0.008
    max_feat = len(x)
    num_feat = np.sum(x == 1)
    error = float('inf')  # initialize error to a default value

    if num_feat == 0 or num_feat == 1:
        cost = 1
    else:
        error = error_rate(xtrain, ytrain, x, opts)
        
        
        pearson_correlation = calculate_pearson_correlation(xtrain[:, x == 1])
        redundancy = np.std(pearson_correlation)  
        
        gamma = 0.002
        cost = alpha * error + beta * (num_feat / max_feat) + gamma * redundancy

    return cost, num_feat, error
