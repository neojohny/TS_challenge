import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


from scipy.spatial.distance import cdist 
from silhouette import silhouette

"""
untility funcitons for data clearning, feature engineering and visulizations
"""


def model_compare(models, X_train, y_train):#, cv = TimeSeriesSplit(5), scoring = rmse_scoring):
    """
    compare metrics between different models
    """
    results = {}
    for m in models:
        results[type(m).__name__] = np.mean(cross_val_score(m, X_train,y_train))#, cv=cv, scoring=rmse_scoring))
    return pd.DataFrame([results])


def get_agg_merchants(df):
    start =  df.groupby('merchant')['time'].min().to_frame('start_time')
    end = df.groupby('merchant')['time'].max().to_frame('end_time')
    counts = df.groupby('merchant')['time'].count().to_frame('transaction_counts')
    time_diff = df.groupby('merchant')['time_diff'].mean().to_frame('time_btween')
    total_sales = df.groupby('merchant')['amount_usd_in_cents'].sum().to_frame('total_sales')
    sales_per = df.groupby('merchant')['amount_usd_in_cents'].mean().to_frame('sales_pertrans')
    df['hours'] = df.time.apply(lambda x:x.time().hour)
    time_of_day = df.groupby('merchant')['hours'].mean().to_frame('time_of_day')
    #daily_trans = df.apply(lambda x:x.counts/(x.end_time-x.start_time).days)
    results = start.join(end).join(counts).join(time_diff).join(total_sales).join(sales_per).join(time_of_day)
    return results.sort_values('transaction_counts',ascending=False)


def feature_eng(agg_mer):
    agg_mer['total_days'] = (agg_mer.end_time-agg_mer.start_time).apply(lambda x:x.days)+1
    agg_mer['trans_perday'] = agg_mer.transaction_counts/agg_mer.total_days
    agg_mer['sales_perday'] = agg_mer.total_sales/agg_mer.total_days
    return agg_mer



def kmeans_elbow(K,X_stan,seed=0):
    distortions = [] 
    inertias = [] 
    silhouette_avg = []
    mapping1 = {} 
    mapping2 = {} 

    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k,random_state=seed).fit(X_stan) 

        distortions.append(sum(np.min(cdist(X_stan, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / X_stan.shape[0]) 
        inertias.append(kmeanModel.inertia_) 
        silhouette_avg.append(silhouette_score(X_stan, kmeanModel.predict(X_stan)))

    return silhouette_avg,distortions,inertias


def inverse_cdf(lam, confi=0.9):
    return -np.log(1-confi)*lam


def get_time_of_day(time):
    #aa = time.apply(lambda x:x.time().hour)
    b = [0,4,8,12,16,20,24]
    l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
    return pd.cut(time, bins=b, labels=l, include_lowest=True)