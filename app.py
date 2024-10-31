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
from sklearn.model_selection import cross_val_score
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

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from xgboost import XGBClassifier
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import math as mt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as ltb

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

from MGWO import MGWO

import pandas as pd
from google_play_scraper import Sort, reviews_all
import base64

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
                  "▪️ Clustering", "▪️ Classification Task (1) - Wrapper based", "▪️ Classification Task (2) - Filtering based",
                  "▪️ Scrapper : Comments from play store",
                  "▪️ Optimizer", "▪️ Classical Optimization Methods"]
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

    elif choice == "▪️ Classification Task (1) - Wrapper based":
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
                
    elif choice == "▪️ Classification Task (2) - Filtering based":
        st.subheader("Classification using multiple types of models and multiple strategies.")   
        st.text("Please ensure that your target variable is named 'Class,' and that your features are already in numeric format.")
        data = st.file_uploader("Please put your data here!", type = ["csv", "txt", "xls"])
        if data is not None:
            df = pd.read_csv(data)
            #df.dropna(inplace=True)
            # st.dataframe(df.head())
            st.success("Your data has been uploaded successfully!")
            all_columns = df.columns.to_list()
            #selected_columns = st.multiselect("Select the variable to be reviewed", all_columns)
            if len(all_columns) > 0:
                class_column = 'Class'
                new_df = df[all_columns]
                st.dataframe(new_df)
                
                sns.set_style("whitegrid")
                
                plt.figure(figsize=(8, 6))
                
     
                sns.countplot(data=new_df, x=class_column, palette='Set2')
                
                plt.title('Distribution of Classes')
                plt.xlabel('Class')
                plt.ylabel('Count')

                class_labels = new_df[class_column].unique()
                
                plt.xticks(ticks=range(len(class_labels)), labels=class_labels)
                st.pyplot() 
                
                fig, ax = plt.subplots(figsize=(15, 15))
    
                sns.heatmap(new_df.corr(), ax=ax, annot=True, linewidth=.6)
                st.pyplot()
                
                #Modeling  part
                X1 = new_df.drop('Class',axis=1)
                y1 = new_df['Class']
                
                # Single classifier based models
                nb = GaussianNB()
                dt = DecisionTreeClassifier()
                svmc = svm.SVC(kernel='poly', degree=2)
                lr = LogisticRegression(random_state = 0)
                knn = KNeighborsClassifier(n_neighbors=7)
                rf = RandomForestClassifier(max_depth= 5, n_estimators= 500)
                extr = ExtraTreesClassifier(max_depth= 10, n_estimators= 500)
                lgb = ltb.LGBMClassifier()
                xgb = XGBClassifier()
                adb = AdaBoostClassifier()

                clf=[nb, dt, svmc, lr, knn, rf, extr, lgb, xgb, adb]
                
                from sklearn.metrics import precision_score
                from sklearn.metrics import recall_score
                
                k = 10
                kf = StratifiedKFold(n_splits=k, random_state=None)
                ac1=[]
                fs1=[]
                rec1=[]
                prec1=[]
                roc1=[]
                for i in clf:
                    acc_score = []
                    fscore = []
                    pr_score = []
                    re_score = []
                    roc_score = []
                    for train_index , test_index in kf.split(X1,y1):
                        X_train , X_test = X1.iloc[train_index,:],X1.iloc[test_index,:]
                        y_train , y_test = y1.iloc[train_index] , y1.iloc[test_index]

                        i.fit(X_train,y_train)
                        pred_values = i.predict(X_test)
                        acc = accuracy_score(pred_values , y_test)
                        acc_score.append(acc)

                        f = f1_score(pred_values , y_test)
                        fscore.append(f)

                        pr = precision_score(pred_values , y_test)
                        pr_score.append(pr)

                        re = recall_score(pred_values , y_test)
                        re_score.append(re)

                        roc = roc_auc_score(pred_values , y_test)
                        roc_score.append(roc)

                    avg_acc_score = sum(acc_score)/k
                    avg_f1_score = sum(fscore)/k
                    avg_pr_score = sum(pr_score)/k
                    avg_re_score = sum(re_score)/k
                    avg_roc_score = sum(roc_score)/k
                    ac1.append(round(avg_acc_score*100,2))
                    fs1.append(round(avg_f1_score*100,2))
                    rec1.append(round(avg_re_score*100,2))
                    prec1.append(round(avg_pr_score*100,2))
                    roc1.append(round(avg_roc_score*100,2))      
                    
                st.success("Results from individual models without strategies")          
                
                clfs =['Naive Bayes', 'Decision Tree', 'Support Vector Machine Classifier', 
                       'Logistic Regression', 'K Nearest Neigbour', 'Random Forest', 
                       'ExtraTreesClassifier', 'LightGBM', 'XGBoost', 'AdaBoost']
                df_result = pd.DataFrame({
                        'Classifier': clfs,
                        'Accuracy in (%)': ac1,
                        'F1 Score in (%)': fs1,
                        'Recall in (%)': rec1,
                        'Precision in (%)': prec1,
                        'ROC AUC in (%)': roc1
                    })
                
                st.dataframe(df_result.sort_values(by='F1 Score in (%)', ascending=False))   
                
                
                st.success("Modelling with balance strategies (K Means SMOTE) and Filtering based")   
        
                X = new_df.drop('Class',axis=1)
                y = new_df['Class'] 
                
                from imblearn.over_sampling import KMeansSMOTE
                smote_kmeans = KMeansSMOTE()
                X, y = smote_kmeans.fit_resample(X, y)
                df3 = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
                
                df = df3
                class_counts = df[class_column].value_counts()
                
                plt.figure(figsize=(8, 6))
                sns.barplot(x=class_counts.index, y=class_counts.values)
                

                plt.title('Class Distribution')
                plt.xlabel('Class')
                plt.ylabel('Count')

                if len(class_counts.index) <= 2:  # Binary classification
                    plt.xticks(ticks=class_counts.index, labels=['Negative', 'Positive'])
                else:  # Multiclass classification
                    plt.xticks(ticks=range(len(class_counts.index)), labels=class_counts.index)
                
                st.pyplot()
                
                X = df.drop('Class',axis=1)
                
                y = df['Class']
                
                from sklearn.ensemble import RandomForestRegressor
                from boruta import BorutaPy

                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

                # Initialize Boruta
                feat_selector = BorutaPy(
                    verbose=2,
                    estimator=model,
                    n_estimators='auto',
                    max_iter=10,
                    random_state=42,
                )

                # Train Boruta
                np.int = np.int32
                np.float = np.float64
                np.bool = np.bool_
                feat_selector.fit(np.array(X), np.array(y))

                # Print support and ranking for each feature
                st.write("\n------Support and Ranking for each feature------\n")
                for i in range(len(feat_selector.support_)):
                    if feat_selector.support_[i]:
                        st.write("Passes the test:", X.columns[i],
                                " - Ranking:", feat_selector.ranking_[i], "✔️")
                    else:
                        st.write("Doesn't pass the test:", X.columns[i],
                                " - Ranking:", feat_selector.ranking_[i], "❌")

                # Features selected by Boruta
                X_filtered = feat_selector.transform(np.array(X))

                st.write("\n------Selected Features------\n")
                selected_features = df.columns[:-1][feat_selector.support_]
                st.write(selected_features)

                # Train the model
                model.fit(X_filtered, y)

                # Compute predictions
                predictions = model.predict(X_filtered)

                # Create a dataframe with real predictions and values
                p = pd.DataFrame({'pred': predictions, 'observed': y})

                # Print the dataframe
                st.write("\n------Predictions and real values------\n")
                st.write(p)

                # Compute RMSE
                mse = ((p['pred'] - p['observed']) ** 2).mean()
                rmse = np.sqrt(mse)
                st.write("\n------RMSE------\n", round(rmse, 3))

                X1 = df[selected_features]
                
                y1 = df['Class']
                
                # Single classifier based models
                nb = GaussianNB()
                dt = DecisionTreeClassifier()
                svmc = svm.SVC(kernel='poly', degree=2)
                lr = LogisticRegression(random_state = 0)
                knn = KNeighborsClassifier(n_neighbors=7)
                rf = RandomForestClassifier(max_depth= 5, n_estimators= 500)
                extr = ExtraTreesClassifier(max_depth= 10, n_estimators= 500)
                lgb = ltb.LGBMClassifier()
                xgb = XGBClassifier()
                adb = AdaBoostClassifier()

                clf=[nb, dt, svmc, lr, knn, rf, extr, lgb, xgb, adb]   
                
                k = 10
                kf = StratifiedKFold(n_splits=k, random_state=None)
                ac2=[]
                fs2=[]
                rec2=[]
                prec2=[]
                roc2=[]
                for i in clf:
                    acc_score = []
                    fscore = []
                    pr_score = []
                    re_score = []
                    roc_score = []
                    for train_index , test_index in kf.split(X1,y1):
                        X_train , X_test = X1.iloc[train_index,:],X1.iloc[test_index,:]
                        y_train , y_test = y1.iloc[train_index] , y1.iloc[test_index]

                        i.fit(X_train,y_train)
                        pred_values = i.predict(X_test)
                        acc = accuracy_score(pred_values , y_test)
                        acc_score.append(acc)

                        f = f1_score(pred_values , y_test)
                        fscore.append(f)

                        pr = precision_score(pred_values , y_test)
                        pr_score.append(pr)

                        re = recall_score(pred_values , y_test)
                        re_score.append(re)

                        roc = roc_auc_score(pred_values , y_test)
                        roc_score.append(roc)

                    avg_acc_score = sum(acc_score)/k
                    avg_f1_score = sum(fscore)/k
                    avg_pr_score = sum(pr_score)/k
                    avg_re_score = sum(re_score)/k
                    avg_roc_score = sum(roc_score)/k
                    ac2.append(round(avg_acc_score*100,2))
                    fs2.append(round(avg_f1_score*100,2))
                    rec2.append(round(avg_re_score*100,2))
                    prec2.append(round(avg_pr_score*100,2))
                    roc2.append(round(avg_roc_score*100,2))
                
                st.success("Results from individual models with multiple strategies")          
                
                clfs =['Naive Bayes', 'Decision Tree', 'Support Vector Machine Classifier', 
                       'Logistic Regression', 'K Nearest Neigbour', 'Random Forest', 
                       'ExtraTreesClassifier', 'LightGBM', 'XGBoost', 'AdaBoost']
                df_result = pd.DataFrame({
                        'Classifier': clfs,
                        'Accuracy in (%)': ac2,
                        'F1 Score in (%)': fs2,
                        'Recall in (%)': rec2,
                        'Precision in (%)': prec2,
                        'ROC AUC in (%)': roc2
                    })
                
                st.dataframe(df_result.sort_values(by='F1 Score in (%)', ascending=False))      
    
    elif choice == "▪️ Scrapper : Comments from play store":
        st.subheader("Extracting application reviews from the Play Store")
        # Input for app_id
        app_id = st.text_input("Please input application ID")
        
        # Button for execution
        if st.button("Extract Reviews"):
            if app_id:
                # Extract reviews from Google Play Store
                result = reviews_all(
                    app_id,
                    sleep_milliseconds=0,
                    lang='en',
                    country='us',
                    sort=Sort.MOST_RELEVANT,
                    filter_score_with=None
                )
                
                # Prepare DataFrame
                reviewid = []
                username = []
                comment = []
                jml_komen_disukai = []
                score = []
                waktu = []
                user_image = []
                
                for i in range(len(result)):
                    reviewid.append(result[i]['reviewId'])
                    username.append(result[i]['userName'])
                    comment.append(result[i]['content'])
                    jml_komen_disukai.append(result[i]["thumbsUpCount"])
                    score.append(result[i]["score"])
                    waktu.append(result[i]["at"])
                    user_image.append(result[i]["userImage"])
                
                df = pd.DataFrame(list(zip(reviewid, user_image, username, comment, jml_komen_disukai, score, waktu)),
                                  columns=['review_id', 'user_image', 'username', 'comment', 'likes_count', 'rating', 'time'])
                
                # Rename columns
                df.columns = ['review_id', 'user_image', 'username', 'comment', 'likes_count', 'rating', 'time']
                
                # Display DataFrame
                st.dataframe(df)
                
                # Save DataFrame to CSV
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64 (bytes), decode to string
                
                # Create a download link with styled button
                href = f'<a href="data:file/csv;base64,{b64}" download="reviews.csv"><button style="background-color:#4CAF50;border:none;color:white;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:12px;">Download CSV</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                
    elif choice == "▪️ Optimizer":
        
        # Title of the application
        st.title("MGWO Toolbox")

        # Sidebar for parameter input
        st.sidebar.header("Parameters")

        search_agents = st.sidebar.number_input("Search Agents", min_value=1, value=30, step=1)
        max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, value=100, step=1)
        lower_bound = st.sidebar.number_input("Lower Bound", value=-10.0)
        upper_bound = st.sidebar.number_input("Upper Bound", value=10.0)
        dimension = st.sidebar.number_input("Dimension", min_value=1, value=30, step=1)

        # Text area for user-defined fitness function
        st.sidebar.subheader("Objective Function")
        objective_function_code = st.sidebar.text_area("Enter your objective function code here:", value="""def user_defined_fitness(x):
            return np.sum(x**2)
        """)

        # Define the user-defined fitness function
        exec(objective_function_code)

        # Optimization process
        if st.sidebar.button("Optimize"):
            # Get the user-defined fitness function
            fitness_function = eval("user_defined_fitness")
            
            best_fitness, best_solution, fitness_history, trajectories, position_history, exploration, exploitation = MGWO(
                search_agents, max_iterations, lower_bound, upper_bound, dimension, fitness_function)

            st.subheader("Optimization Results")

            st.write(f"Best Fitness: {best_fitness}")
            st.write(f"Best Solution: {best_solution}")

            # Plot fitness history
            st.subheader("Fitness History")
            plt.figure(figsize=(10, 6))
            plt.plot(fitness_history)
            plt.xlabel("Iterations")
            plt.ylabel("Fitness")
            st.pyplot(plt)

            # Plot exploration and exploitation
            st.subheader("Exploration and Exploitation")
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_iterations + 1), exploration, label='Exploration %')
            plt.plot(range(1, max_iterations + 1), exploitation, label='Exploitation %')
            plt.xlabel('Iterations')
            plt.ylabel('Percentage')
            plt.legend()
            st.pyplot(plt)    
            
    elif choice == "▪️ Classical Optimization Methods":
        
if __name__ == "__main__":
     main()   
