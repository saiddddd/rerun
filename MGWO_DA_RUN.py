import numpy as np
import pickle
import pandas as pd
from MGWO_DA import jfs
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import datetime
import time
import streamlit as st
import base64

warnings.filterwarnings("ignore")

encoder = LabelEncoder()

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2
    
def optimized_combined_kernel(X, Y):
    # Kernel Linier
    K_lin = np.dot(X, Y.T)
    
    # Kernel RBF dengan gamma yang dioptimalkan
    gamma = 0.2
    squared_distance = np.sum(X**2, 1).reshape(-1, 1) + np.sum(Y**2, 1) - 2 * np.dot(X, Y.T)
    K_rbf = np.exp(-gamma * squared_distance)
    
    # Gabungan kernel linier dan RBF
    beta = 0.001
    return beta * K_lin + (1 - beta) * K_rbf

def fast_fitness_evaluation(X_subset, y_subset, sf):
    mdl = SVC(kernel=optimized_combined_kernel, random_state=50, probability=True)
    ss = StratifiedKFold(n_splits=5, shuffle=True)
    f1_scores, auc_scores, acc_scores = [], [], []
    
    for train, test in ss.split(X_subset, y_subset): 
        mdl.fit(X_subset[train][:, sf], y_subset[train])
        y_predict = mdl.predict(X_subset[test][:, sf])
        y_proba = mdl.predict_proba(X_subset[test][:, sf])
        score_f1 = f1_score(y_subset[test], y_predict, average='weighted')
        score_acc = accuracy_score(y_subset[test], y_predict)
        
        if len(np.unique(y_subset)) == 2:  # Biner
            y_proba = y_proba[:, 1]
            score_auc = roc_auc_score(y_subset[test], y_proba)
        else:
            score_auc = roc_auc_score(y_subset[test], y_proba, multi_class='ovo', average='weighted')
        
        f1_scores.append(score_f1)
        auc_scores.append(score_auc)
        acc_scores.append(score_acc)
    
    return np.mean(f1_scores), np.mean(auc_scores), np.mean(acc_scores), mdl

def feature_select(data, particle_num=[10], iteration_num=[1]):
    start = time.time()
    st.write('this process started at: ', datetime.datetime.now())

    data = data
    particle = particle_num
    iteration = iteration_num

    st.write('shape of dataset:', data.shape)
    X = data.loc[:, data.columns != 'Class'].to_numpy()
    y = data['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = X_scaled

    y_encoded = encoder.fit_transform(y)
    y = y_encoded.astype(int)

    feat = X
    label = y

    final_model = None

    for n in particle:
        num_feature_list = []
        f1_list = []
        auc_list = []
        acc_list = []
        best_sf = None  # Menyimpan indeks fitur terpilih terbaik
        best_fitness = 0.0  # Menyimpan fitness terbaik
        solution_list = []

        subset_size = int(1.0 * len(y))
        random_indices = np.random.choice(len(y), subset_size, replace=False)
        X_subset = X[random_indices]
        y_subset = y[random_indices]
        
        with st.spinner(f'Running for {n} particles with {iteration[0]} max iterations...'):
            status_placeholder = st.empty()

            for j in iteration:
                for i in range(10):
                    status_placeholder.write(f'For particle {n} with {j} max iterations, now in {i} iteration(s)')
                    fold = {'x': feat, 'y': label}
                    k = 5
                    N = n
                    T = j
                    opts = {'k': k, 'fold': fold, 'N': N, 'T': T}

                    fmdl = jfs(feat, label, opts)
                    sf = fmdl['sf']  # Perlu deklarasi sf di sini
                    num_feat = fmdl['nf']
                    num_feature_list.append(num_feat)

                    f1, auc, acc, mdl = fast_fitness_evaluation(X_subset, y_subset, sf)
                    f1_list.append(f1)
                    auc_list.append(auc)
                    acc_list.append(acc)

                    # Periksa apakah fitness saat ini lebih baik dari yang sebelumnya
                    current_fitness = np.mean(f1_list)  # Misalnya, menggunakan F1-score sebagai fitness
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_sf = sf.copy()  # Simpan indeks fitur terpilih saat ini

                status_placeholder.empty()

                solution_data = {
                    'iteration': j,
                    'number of features': len(best_sf),
                    'best F1-score': max(f1_list),
                    'median F1-score': mediannum(f1_list),
                    'std F1-score': np.std(f1_list),
                    'best AUC-score': max(auc_list),
                    'median AUC-score': mediannum(auc_list),
                    'std AUC-score': np.std(auc_list),
                    'best Accuracy': max(acc_list),
                    'median Accuracy': mediannum(acc_list),
                    'std Accuracy': np.std(acc_list),
                    'selected features': best_sf  # Tambahkan indeks fitur terpilih terbaik
                }

                solution_list.append(solution_data)
                
                # Simpan model terakhir
                final_model = {'particle': n, 'iteration': j, 'model': mdl, 'selected_features': best_sf}

        st.write(solution_list)        
        st.write("Finish processing in ", time.time() - start, 's')
        st.write('This process finished at: ', datetime.datetime.now())

    # Pastikan ada model yang tersimpan sebelum mencoba menyimpannya
    if final_model and final_model['model']:
        # Simpan model ke dalam file pickle
        final_model_filename = "final_model.pkl"
        with open(final_model_filename, 'wb') as final_model_file:
            pickle.dump(final_model, final_model_file)

        # Tambahkan tombol download untuk file pickle model terakhir
        download_link = create_download_link(final_model_filename)
        st.markdown(download_link, unsafe_allow_html=True)
        st.success("Model has been successfully trained and saved!")

def create_download_link(filename):
    with open(filename, 'rb') as file:
        data = file.read()
        base64_encoded = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{base64_encoded}" download="{filename}">Click here to download the final model</a>'
    return href

# Contoh pemanggilan:
# feature_select("cleavland.csv")
