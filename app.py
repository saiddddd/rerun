#Core pckg
import os
import streamlit as st
#import base64
#from io import BytesIO

#EDA pckgs
import pandas as pd
import numpy as np

#Data Viz pckgs
import seaborn as sns
import matplotlib.pyplot as plt
import  matplotlib
matplotlib.use('Agg')

#ML pckgs
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn import model_selection
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from  sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.neighbors import KNeighborsClassifier
from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from  sklearn.svm import  SVC
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
import pylab as pl
import codecs
from sklearn.preprocessing import StandardScaler

from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv 

import random
import time
import math

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import benchmarks
from sklearn.preprocessing import LabelEncoder

st.set_option('deprecation.showfileUploaderEncoding', False)


def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)

def main():
    st.title("Welcome to My Site ⚜️")
    st.text("Part of my activity during my work.")
    st.image('danbo.jpg', use_column_width=True)
    aktivitas = ["▪️ About","▪️ EDA 1", "▪️ EDA 2",
                  #"▪️ Modelling", "▪️ Fraud Detection", 
                  "▪️ Clustering", "▪️ Classification Task"]
    choice = st.sidebar.selectbox("Select your activity here", aktivitas)
    if choice == "▪️ About":
        st.subheader("About 🧬")   
        st.success("This work is aim to help us to show our model's result to the audience.") 
        st.warning("Please give suggestions if you have any issues for this work, thank you!")
    elif choice == "▪️ EDA 1":
        st.subheader("Exploratory Data Analysis (1)")
        data = st.file_uploader("Please upload your data here :)", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            if st.checkbox("Your Data Size"):
                st.write(df.shape)
            if st.checkbox("Variables/Features"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            if st.checkbox("Please select certain variable(s)/feature(s)"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Choose", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
            if st.checkbox("Basic Description"):
                st.success("First observation!")
                st.write(df.describe(include='all'))

                def overview_data(data, features):
                    print(f"Dataset Shape: {data.shape}")
                    
                    df = data[features]
                    #df['target'] = data[target]
                    overview = pd.DataFrame(df.dtypes,columns=['dtypes'])
                    overview = overview.reset_index()
                    overview['features'] = overview['index']
                    overview = overview[['features','dtypes']]
                    overview['Missing'] = df.isnull().sum().values   
                    overview['%Missing'] = df.isnull().sum().values/df.shape[0]
                    overview['%Missing'] = overview['%Missing'].apply(lambda x: format(x, '.2%'))
                    overview['Uniqueness'] = df.nunique().values
                    overview['%Unique'] = df.nunique().values/df.shape[0]
                    overview['%Unique'] = overview['%Unique'].apply(lambda x: format(x, '.2%'))
                    
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Minumum'] = df[var].min()
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Quantile 1'] = np.nanpercentile(df[var], 25)  
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Median'] = df[var].median()  
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Mean'] = df[var].mean()    
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Quartile 3'] = np.nanpercentile(df[var], 75)
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Maximum'] = df[var].max()
                    for var in overview['features']:
                        overview.loc[overview['features'] == var, 'Standar Deviation'] = df[var].std()    
                    return overview
                st.text("Please select certain numerical variable(s)/feature(s)")
                columns = df.columns.to_list()
                selected_columns = st.multiselect("Selected column(s)", columns)
                new_df = df[selected_columns]
                st.success("Second observation!")
                st.write(overview_data(new_df, selected_columns))

            if st.checkbox("Target class"):
                st.text("Note that if your dataset has a target column, make sure it is positioned on the far right.")
                st.write(df.iloc[:,-1].value_counts())    
            if st.checkbox("Correlation"):
                plt.subplots(figsize=(10,10))
                st.write(sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True))
                st.pyplot()
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Please select 1 variable/feature", all_columns)
                if len(columns_to_plot) != 0:
                    pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie_plot)
                    st.pyplot()
            if st.checkbox("Scatter Plot"):
                all_columns = df.columns.to_list()
                if len(all_columns) != 0:
                    # pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    # st.write(pie_plot)
                    # st.pyplot()
                    x_variable = st.selectbox("Please select first variable/feature", all_columns)
                    y_variable = st.selectbox("Please select second variable/feature", all_columns)

                    # Buat scatter plot dengan Matplotlib
                    plt.figure(figsize=(8, 6))
                    plt.scatter(df[x_variable], df[y_variable],
                                 #c=data['kolom_label'], 
                                 cmap='viridis')
                    plt.title(f'Scatter Plot for {x_variable} and {y_variable}')
                    plt.xlabel(x_variable)
                    plt.ylabel(y_variable)
                    #plt.colorbar(label='kolom_label')
                    st.pyplot()

    elif choice == "▪️ EDA 2":
        st.subheader("Data Visualization")
        data = st.file_uploader("Please upload your data here :)", type = ["csv", "txt", "xls"])
        
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            all_columns_names = df.columns.to_list()
            if len(all_columns_names) > 0:
                type_of_plot = st.selectbox("Select the type of graph to be displayed", ["area","bar","line","hist",
                                                                                         "another distribution plot",
                                                                                         "box","kde"])
                selected_column_names = st.multiselect("Select the variable/Feature to be plotted", all_columns_names)
                if st.button("Generate Plot"): #bagian button
                    st.success("The graph of the {} selection for the {} data will be displayed immediately".format(type_of_plot,selected_column_names))
                    if type_of_plot == "area":
                        cust_data = df[selected_column_names]
                        st.area_chart(cust_data)
                    elif type_of_plot == "bar":
                        cust_data = df[selected_column_names]
                        st.bar_chart(cust_data)
                    elif type_of_plot == "line":
                        cust_data = df[selected_column_names]
                        st.line_chart(cust_data)
                    elif type_of_plot == "another distribution plot":
                        ax = sns.histplot(data=df[selected_column_names], bins=20, stat='density', alpha= 1, kde=True,
                        edgecolor='white', linewidth=0.5,
                        line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
                        ax.get_lines()[0].set_color('black')
                        ax.legend(frameon=False)
                        ax.set_title(f'The Distribution of {selected_column_names}', fontsize=14, pad=15)
                        #st.write(ax)
                        st.pyplot()  
                    elif type_of_plot:
                        cust_plot = df[selected_column_names].plot(kind=type_of_plot)
                        st.write(cust_plot)
                        st.pyplot()      
    # elif choice == "▪️ Modelling":
    #     st.subheader("Machine Learning Model")
    #     status = st.radio("Choose your data availability: ", ["From Sklearn", "From Your Database"])
    #     if status == "From Sklearn":
    #         st.success("You choose dataset from sklearn!")
    #         dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
    #         classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))
    #         def get_dataset(dataset_name):
    #             if dataset_name == "Iris":
    #                 data = datasets.load_iris()
    #             elif dataset_name == "Breast Cancer":
    #                 data = datasets.load_breast_cancer()
    #             else:
    #                 data = datasets.load_wine()
    #             X = data.data
    #             y = data.target
    #             return X, y   

    #         X, y = get_dataset(dataset_name)
    #         st.write("shape of dataset", X.shape)
    #         st.write("number of classes", len(np.unique(y)))

    #         def add_paramater_ui(clf_name):
    #             params = dict()
    #             if clf_name == "KNN":
    #                 K = st.sidebar.slider("K", 1, 15)
    #                 params["K"] = K
    #             elif clf_name == "SVM":
    #                 C = st.sidebar.slider("C", 0.01, 10.0)
    #                 params["C"] = C
    #             else:
    #                 max_depth = st.sidebar.slider("max_depth", 2, 15)
    #                 n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    #                 params["max_depth"] = max_depth
    #                 params["n_estimators"] = n_estimators
    #             return params   
    #         params = add_paramater_ui(classifier_name)

    #         def get_classifier(clf_name, params):
    #             if clf_name == "KNN":
    #                 clf = KNeighborsClassifier(n_neighbors = params["K"])
    #             elif clf_name == "SVM":
    #                 clf = SVC(C = params["C"])
    #             else:
    #                 clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 2)
    #             return clf

    #         clf = get_classifier(classifier_name, params)    

    #         #Classfication
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    #         acc = accuracy_score(y_test, y_pred)
    #         st.write(f"classifer = {classifier_name}")
    #         st.write(f"accuracy = {acc}")

    #         #PLOT
    #         pca = PCA(2)
    #         X_projected = pca.fit_transform(X)

    #         x1 = X_projected[:,0]
    #         x2 = X_projected[:,1]
    #         fig = plt.figure()
    #         plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = "viridis")
    #         plt.xlabel("First component")
    #         plt.ylabel("Second component")
    #         plt.colorbar()
    #         st.pyplot()

    #     else:
    #         data = st.file_uploader("Please upload your clean data here :)", type = ["csv", "txt", "xls"])
            
    #         if data is not None:
    #             df = pd.read_csv(data)
    #             st.dataframe(df.head())
    #             st.success("Your data has been uploaded successfully!")
    #             st.error("Please make sure that your target class is the last column!")
    #             X = df.iloc[:,0:-1]
    #             Y = df.iloc[:,-1]
    #             y = Y
    #             seed = 7

    #             models = []
    #             models.append(("LR", LogisticRegression()))
    #             models.append(("LDA", LinearDiscriminantAnalysis()))
    #             models.append(("KNN", KNeighborsClassifier()))
    #             models.append(("CART", DecisionTreeClassifier()))
    #             models.append(("NB", GaussianNB()))
    #             models.append(("SVM", SVC()))

    #             model_names = []
    #             model_mean = []
    #             model_std = []
    #             all_models = []
    #             scoring = 'accuracy'
    #             for name,model in models:
    #                 kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #                 cv_results = model_selection.cross_val_score(model, X,Y,cv=kfold, scoring = scoring)
    #                 model_names.append(name)
    #                 model_mean.append(cv_results.mean())
    #                 model_std.append(cv_results.std())
    #                 accuracy_results = {"model":name, "akurasi model": cv_results.mean(), "standar deviasi": cv_results.std()}
    #                 all_models.append(accuracy_results)
    #             if  st.checkbox("Tabel Metrik Model"):
    #                 st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std), columns=["Model", "Akurasi", "Standar Deviasi"]))
    #             if st.checkbox("Metrik Model Json"):
    #                 st.json(all_models)    
    #             if st.checkbox("Details"):
    #                 st.subheader("Models Detail For Classification") 
    #                 classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))
    #                 st.write("shape of dataset", X.shape)
    #                 st.write("number of classes", len(np.unique(y)))
    #                 def add_paramater_ui(clf_name):
    #                     params = dict()
    #                     if clf_name == "KNN":
    #                         K = st.sidebar.slider("K", 1, 15)
    #                         params["K"] = K
    #                     elif clf_name == "SVM":
    #                         C = st.sidebar.slider("C", 0.01, 10.0)
    #                         params["C"] = C
    #                     else:
    #                         max_depth = st.sidebar.slider("max_depth", 2, 15)
    #                         n_estimators = st.sidebar.slider("n_estimators", 1, 100)
    #                         params["max_depth"] = max_depth
    #                         params["n_estimators"] = n_estimators
    #                     return params   
    #                 params = add_paramater_ui(classifier_name)
    #                 def get_classifier(clf_name, params):
    #                     if clf_name == "KNN":
    #                         clf = KNeighborsClassifier(n_neighbors = params["K"])
    #                     elif clf_name == "SVM":
    #                         clf = SVC(C = params["C"])
    #                     else:
    #                         clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 2)
    #                     return clf
    #                 clf = get_classifier(classifier_name, params)    

    #                 #Classfication
    #                 test_size = st.sidebar.slider("test_size", 0.05, 0.30)
    #                 random_state = st.sidebar.slider("random_state", 0, 9)
    #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #                 clf.fit(X_train, y_train)
    #                 y_pred = clf.predict(X_test)
    #                 acc = accuracy_score(y_test, y_pred)
    #                 st.write(f"classifer = {classifier_name}")
    #                 st.write(f"accuracy = {acc}")
    #                 #PLOT
    #                 pca = PCA(2)
    #                 X_projected = pca.fit_transform(X)
    #                 x1 = X_projected[:,0]
    #                 x2 = X_projected[:,1]
    #                 fig = plt.figure()
    #                 plt.scatter(x1, x2, alpha = 0.8, cmap = "viridis")
    #                 plt.xlabel("komponen pertama")
    #                 plt.ylabel("komponen kedua")
    #                 plt.colorbar()
    #                 st.pyplot()

    # elif choice == "▪️ Fraud Detection":
    #     st.subheader("Fraud? Really?")   
    #     data = st.file_uploader("Where is your data?", type = ["csv", "txt", "xls"])
    #     if data is not None:
    #         df = pd.read_csv(data)
    #         st.dataframe(df.head())
    #         st.success("Your data has been uploaded successfully!")
    #         all_columns = df.columns.to_list()
    #         if st.checkbox("Pilih Variable/Feature yang akan ditinjau"):
    #             selected_columns = st.multiselect("Pilih", all_columns)
    #             pjg = len(selected_columns)
    #             delta = df[selected_columns]
    #             st.dataframe(delta)
    #             if st.checkbox("Ukuran data"):
    #                 st.write(delta.shape)
    #             if st.checkbox("Basic Description"):
    #                 st.write(delta.describe())
    #             if st.checkbox("Go to Fraud"):    
    #                 from numpy.linalg import inv
    #                 meanValue = delta.mean()
    #                 covValue = delta.cov()
    #                 X = delta.to_numpy()
    #                 S = covValue.to_numpy()
    #                 for i in range(pjg):
    #                     X[:,i] = X[:,i] - meanValue[i] 
    #                 def mahalanobis(row):
    #                     return np.matmul(row,S).dot(row)
    #                 anomaly_score = np.apply_along_axis( mahalanobis, axis=1, arr=X)    
    #                 anom = pd.DataFrame(anomaly_score, columns=['Anomaly score'])
    #                 result = pd.concat((delta,anom), axis=1)
    #                 result['norek_penerima'] = df['norek_penerima']
    #                 def keterangan(x):
    #                     if x > 1000000:
    #                         return "Tidak Wajar"
    #                     else:
    #                         return "Wajar"
    #                 result['Keputusan'] = result['Anomaly score'].apply(keterangan)
    #                 st.dataframe(result)
    #                 value = result['Keputusan'].unique().tolist()
    #                 if st.checkbox("Lihat yang tidak wajar?"):
    #                     delta_new = result[result['Keputusan']=="Tidak Wajar"]
    #                     st.dataframe(delta_new)
    #                 st.subheader("Bagaimana dengan komposisi keputusan?")
    #                 if st.checkbox("Look at the Pie Chart"):
    #                     all_columns = result.columns.to_list()
    #                     columns_to_plot = st.selectbox("Sila pilih 1 variabel", all_columns)
    #                     pie_plot_dua = result[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
    #                     st.write(pie_plot_dua)
    #                     st.pyplot()
    elif choice == "▪️ Clustering":
        st.subheader("Clustering with K-Means")   
        st.text("Since the method used here is based on K-Means clustering, make sure your dataset is all numeric.")
        data = st.file_uploader("Please put your data here!", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            #df.dropna(inplace=True)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            all_columns = df.columns.to_list()
            selected_columns = st.multiselect("Select the variable to be reviewed", all_columns)
            if len(selected_columns)>0:
                new_df = df[selected_columns]
                st.dataframe(new_df)
                X = new_df.values[:,]
                X = np.nan_to_num(X)
                Clus_dataSet = StandardScaler().fit_transform(X)
                st.text("Please determine the number of clusters before determining the optimal number of clusters!")
                clusterNum = st.slider("Number of clusters", 1, 5)
                random_state = st.slider("Random state", 0, 5)
                if clusterNum > 1:
                    k_means = KMeans(n_clusters = clusterNum, init = "k-means++", random_state=random_state)#init = "k-means++",  
                    k_means.fit(X)
                    labels = k_means.labels_
                    new_df["Clus_km"] = labels
                    if st.button("Lihat hasilnya"):
                        st.dataframe(new_df)
                        #df = new_df # your dataframe
                        #st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                        #st.markdown(get_binary_file_downloader_html('data.csv', 'My Data'), unsafe_allow_html=True)
                    #PLOT
                    st.subheader("How about the visuals?")
                    pca = PCA(2)
                    X_projected = pca.fit_transform(X)
                    x1 = X_projected[:,0]
                    x2 = X_projected[:,1]
                    fig = plt.figure()
                    y = new_df["Clus_km"]
                    plt.scatter(x1, x2, c = y, alpha = 0.8)
                    plt.xlabel("First component")
                    plt.ylabel("Second component")
                    plt.colorbar()
                    st.pyplot()
                    st.subheader("What about the evaluation?")
                    eval_satu = metrics.calinski_harabasz_score(X, labels)
                    eval_satu = round(eval_satu,2)
                    eval_dua = davies_bouldin_score(X, labels)
                    eval_dua = round(eval_dua,2)
                    st.text("Calinski-Harabasz Index")
                    st.warning(eval_satu)
                    st.text("Davies-Bouldin Index")
                    st.warning(eval_dua)
                    st.success("Remember that a lower index Davies-Bouldin index relates to a model with better separation between the clusters and a higher index Calinski-Harabasz (score) relates to a model with better defined clusters i.e The index is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)")

    elif choice == "▪️ Classification Task":
        import MGWO_DA_RUN
        st.subheader("Classification Using MGWO-DA Feature Selection")   
        st.text("Please ensure that your target variable is named 'Class,' and that your features are already in numeric format.")
        data = st.file_uploader("Please put your data here!", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            #df.dropna(inplace=True)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            all_columns = df.columns.to_list()
            #selected_columns = st.multiselect("Select the variable to be reviewed", all_columns)
            if len(all_columns) > 0:
                new_df = df[all_columns]
                st.dataframe(new_df)
                MGWO_DA_RUN.feature_select(new_df)
                
    

if __name__ == "__main__":
     main()   
