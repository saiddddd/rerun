#Core pckg
import os
import streamlit as st

#import base64
#from io import BytesIO

#EDA pckgs
import pandas as pd
import numpy as np

#Data Viz pckgs
# import pygwalker as pyg
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
from pandas_profiling import ProfileReport 
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv 

st.set_option('deprecation.showfileUploaderEncoding', False)

def overview_data(data, features):
    print(f"Dataset Shape: {data.shape}")
    
    df = data[features]
#     df['target'] = data[target]
    
    total_cnt = df.shape[0]
    

    overview = pd.DataFrame(df.dtypes,columns=['dtypes'])
    overview = overview.reset_index()
    overview['features'] = overview['index']
    overview = overview[['features','dtypes']]
    overview['Missing'] = df.isnull().sum().values   
    overview['%Missing'] = df.isnull().sum().values/df.shape[0]
    overview['%Missing'] = overview['%Missing'].apply(lambda x: format(x, '.2%'))
    overview['Uniques'] = df.nunique().values
    overview['%Unique'] = df.nunique().values/df.shape[0]
    overview['%Unique'] = overview['%Unique'].apply(lambda x: format(x, '.2%')) 
        
    for var in overview['features']:
        nan_cnt = np.sum(df[var].isnull())
        zero_cnt = np.sum(df[var] == 0) if df[var].dtypes != 'object' else np.sum(df[var] == '')
        
        overview.loc[overview['features'] == var, 'nan_ratio'] = nan_cnt/total_cnt
        overview.loc[overview['features'] == var, 'zero_ratio'] = zero_cnt/total_cnt
        
        overview.loc[overview['features'] == var, 'coverage'] = df[var].count()/total_cnt
        overview.loc[overview['features'] == var, 'coverage_nonzero'] =  overview.loc[overview['features'] == var, 'coverage'] - overview.loc[overview['features'] == var, 'zero_ratio']
        
    return overview


def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    #components.html(page,width=width,height=height,scrolling=True)

def main():
    st.title("Welcome to My Site ⚜️")
    st.text("Part of my activity during my work.")
    st.image('_124911599_swallowtail_vulnerable_iainhleach.jpg', use_column_width=True)
    aktivitas = ["▪️ About","▪️ EDA", 
                 "▪️ More EDA","▪️ Plotting", 
                 "▪️ Your Interface Graph"
                #  "▪️ Modelling", "▪️ Fraud Detection", "▪️ Clustering","▪️ Stream ML"
                 ]
    choice = st.sidebar.selectbox("Select your activity here", aktivitas)
    if choice == "▪️ About":
        st.subheader("About 🧬")   
        st.success("This work is aim to help us to show or learn or explore the data that we want to check.") 
        st.warning("Please give suggestions if you have any issues for this work, thank you!")
    elif choice == "▪️ EDA":
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Please upload your data here :)", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            if st.checkbox("Ukuran data"):
                st.write(df.shape)
            if st.checkbox("Variables/Features"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            if st.checkbox("Please select certain features or variables..."):
                selected_columns = st.multiselect("Pilih", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
            if st.checkbox("Basic Description"):
                st.write(df.describe())
            if st.checkbox("Independent/Target Class"):
                st.write(df.iloc[:,-1].value_counts())    
            if st.checkbox("Correlation"):
                plt.subplots(figsize=(10,10))
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Please select 1 feature or variable", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
    elif choice == "▪️ More EDA":
        st.subheader("Looking into Further Data Exploration")
        data = st.file_uploader("Where is your data?", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            st.text("Statistic of your data is here...")
            st.success("No. 1")
            overview = overview_data(df, df.columns.values)
            overview
            st.success("No. 2")
            print(df.describe())
            
            status = st.radio("Please choose your detail: ", ["Sweetviz", "Pandas Profiling"])
            if status == "Pandas Profiling":
                st.success("You choose pandas profile to check the data!")
                if st.button("Generate your report"):
                    profile = ProfileReport(df)
                    st_profile_report(profile)
            else:
                st.success("You choose sweetviz to check the data!")
                if st.button("Generate your report"):
                    report = sv.analyze(df)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html")    


    elif choice == "▪️ Plotting":
        st.subheader("Data Visualization")
        data = st.file_uploader("Please upload your data here :)", type = ["csv", "txt", "xls"])
        
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            all_columns_names = df.columns.to_list()
            if len(all_columns_names) > 0:
                type_of_plot = st.selectbox("Select the type of graph to display", ["area","bar","line","hist","box","kde"])
                selected_column_names = st.multiselect("Select the variable/Feature to be plotted", all_columns_names)
                if st.button("Generate Plot"): #bagian button
                    st.success("Graph of the selection {} to data {} will be displayed immediately".format(type_of_plot,selected_column_names))
                    if type_of_plot == "area":
                        cust_data = df[selected_column_names]
                        st.area_chart(cust_data)
                    elif type_of_plot == "bar":
                        cust_data = df[selected_column_names]
                        st.bar_chart(cust_data)
                    elif type_of_plot == "line":
                        cust_data = df[selected_column_names]
                        st.line_chart(cust_data)
                    elif type_of_plot:
                        cust_plot = df[selected_column_names].plot(kind=type_of_plot)
                        st.write(cust_plot)
                        st.pyplot()  
                      
    elif choice == "▪️ Your Interface Graph":
        st.subheader("Looking Further into Data Exploration")
        data = st.file_uploader("Where is your data?", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            st.success("Let's play around!:")
            st.success("Use pygwalker! example here : [link] (https://colab.research.google.com/drive/1-E-xDr5o4k8ohcNEoGsnzlOBqSX0lJeU#scrollTo=zUKCbBnzTB36)")
            
                    
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
    #         plt.xlabel("komponen pertama")
    #         plt.ylabel("komponen kedua")
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
    # elif choice == "▪️ Clustering":
    #     st.subheader("Clustering with K-Means")   
    #     data = st.file_uploader("Please put your data here!", type = ["csv", "txt", "xls"])
    #     if data is not None:
    #         df = pd.read_csv(data)
    #         df.dropna(inplace=True)
    #         st.dataframe(df.head())
    #         st.success("Your data has been uploaded successfully!")
    #         all_columns = df.columns.to_list()
    #         selected_columns = st.multiselect("Pilih variabel yang akan ditinjau", all_columns)
    #         if len(selected_columns)>0:
    #             new_df = df[selected_columns]
    #             st.dataframe(new_df)
    #             X = new_df.values[:,]
    #             X = np.nan_to_num(X)
    #             Clus_dataSet = StandardScaler().fit_transform(X)
    #             st.text("Sila tentukan banyaknya cluster sebelum ditentukan banyak klaster yang optimal!")
    #             clusterNum = st.slider("Banyaknya cluster", 1, 5)
    #             random_state = st.slider("Random state", 0, 5)
    #             if clusterNum > 1:
    #                 k_means = KMeans(n_clusters = clusterNum, init = "k-means++", random_state=random_state)#init = "k-means++",  
    #                 k_means.fit(X)
    #                 labels = k_means.labels_
    #                 new_df["Clus_km"] = labels
    #                 if st.button("Lihat hasilnya"):
    #                     st.dataframe(new_df)
    #                     #df = new_df # your dataframe
    #                     #st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    #                     #st.markdown(get_binary_file_downloader_html('data.csv', 'My Data'), unsafe_allow_html=True)
    #                 #PLOT
    #                 st.subheader("Bagaimana dengan visualnya?")
    #                 pca = PCA(2)
    #                 X_projected = pca.fit_transform(X)
    #                 x1 = X_projected[:,0]
    #                 x2 = X_projected[:,1]
    #                 fig = plt.figure()
    #                 y = new_df["Clus_km"]
    #                 plt.scatter(x1, x2, c = y, alpha = 0.8)
    #                 plt.xlabel("komponen pertama")
    #                 plt.ylabel("komponen kedua")
    #                 plt.colorbar()
    #                 st.pyplot()
    #                 st.subheader("Bagaimana dengan evaluasinya?")
    #                 eval_satu = metrics.calinski_harabasz_score(X, labels)
    #                 eval_satu = round(eval_satu,2)
    #                 eval_dua = davies_bouldin_score(X, labels)
    #                 eval_dua = round(eval_dua,2)
    #                 st.text("Calinski-Harabasz Index")
    #                 st.warning(eval_satu)
    #                 st.text("Davies-Bouldin Index")
    #                 st.warning(eval_dua)
    #                 st.success("Remember that a lower index Davies-Bouldin index relates to a model with better separation between the clusters and a higher index Calinski-Harabasz (score) relates to a model with better defined clusters i.e The index is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)")


if __name__ == "__main__":
     main()   
