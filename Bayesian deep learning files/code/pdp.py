
import pandas as pd
pd.set_option('display.max_columns', 500)

v = pd.read_excel(r"data(7).xlsx")
v_list=[]
for index,name in (enumerate(v['variant'].value_counts().index.tolist())):
    v_list.append(name)
v_list.append('not_sequenced')
month_data_encoded=pd.read_excel(r"month_data_encoded.xlsx")
data_encoded=pd.read_excel(r"data_encoded.xlsx")
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
model = tf.keras.models.load_model(r"convmodel")
def accuracy_score(preds, labels):
    #return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))
    #predictednumbers=np.argmax(preds, axis=1)
    #realnumbers=np.argmax(labels, axis=1)
    diff=np.subtract(preds, labels)
    return np.sqrt(sum(np.power(diff, 2))/len(diff))
month_codes=[]
for index,name in (enumerate(data_encoded['month'].value_counts().index.tolist())):
    #for name in['Germany']:
        month=name
        data_encoded_2=data_encoded.loc[(data_encoded['month'] == month)]
        month_codes.append([month,[data_encoded_2.iloc[0]['month_0'],data_encoded_2.iloc[0]['month_1'] ,data_encoded_2.iloc[0]['month_2'] ,data_encoded_2.iloc[0]['month_3']  ] ])

        
        
from statistics import mean
from Big_list import Big_list
filtered = []

for i in Big_list:
    if mean(i[1]) >= 6.3:
        filtered.append(i)


vv=[[i[0],mean(i[1])] for i in list(filtered)]
def Sort(sub_li):
   
    l = len(sub_li)
     
    for i in range(0, l):
        for j in range(0, l-i-1):
             
            if (sub_li[j][1] > sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j] = sub_li[j + 1]
                sub_li[j + 1] = tempo
     
    return sub_li
ff=Sort(vv)
ff=[ff[i][0] for i in range(0,(len(ff)-2))]



feature_list=ff
xxx_list=[]
cat_nr=100
def valuation_formula(step):
    return step
for index,name in (enumerate(month_data_encoded['countriesAndTerritories'].value_counts().index.tolist())):
    #for name in['Germany']:
        land=name
        print(land)
        month_data_encoded_land=month_data_encoded.loc[(month_data_encoded['countriesAndTerritories'] == land)]
        mde=month_data_encoded_land
        mde=mde.drop(['dateRep', 'cases' , 'Rescd' , '7days_before_mean','7days_after_mean', 'week' ,'year' , 'index', 'month', 'countriesAndTerritories', 'va' , 'vaccin'], axis=1)
        mde = mde.reindex(sorted(mde.columns), axis=1)
        percent_list=[]
        number_list=[-10000]
        

        
        test_origin=mde
        xx_list=[]
        test=test_origin.copy(deep=True)
        for step in [i[1] for i in month_codes]:
            test['month_0'] = test.apply(lambda row: step[0], axis=1)
            test['month_1'] = test.apply(lambda row: step[1], axis=1)
            test['month_2'] = test.apply(lambda row: step[2], axis=1)
            test['month_3'] = test.apply(lambda row: step[3], axis=1)
            
            X_test = test.drop(labels = ["rp_zeitraum"],axis = 1)
            X_test = X_test.values.reshape(-1,1,111,1)
            
            n_mc_run = 100
            #med_prob_thres = 0.35

            y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
            y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
            y_predicted_list=[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(X_test)) ]
            
            mean_predicted=np.mean(y_predicted_list)
            xx_list.append(mean_predicted)
        xxx_list.append([land,'month',[[i[0] for i in month_codes],xx_list]])
        print([land,'month',[[i[0] for i in month_codes],xx_list]])
        print(xx_list)
        for xx in feature_list:
            print(xx)
            xx_list=[]
            test=test_origin.copy(deep=True)
            for step in [0,1.0]:
                test[xx] = test.apply(lambda row: valuation_formula(step), axis=1)
                if xx in v_list:
                    vv_list=v_list.copy()
                    vv_list.remove(xx)
                    for i in vv_list:
                        test[i] = test[i].multiply(1-step)
                
                X_test = test.drop(labels = ["rp_zeitraum"],axis = 1)
                X_test = X_test.values.reshape(-1,1,111,1)
                
                n_mc_run = 100
                #med_prob_thres = 0.35

                y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
                y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
                y_predicted_list=[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(X_test)) ]
                
                mean_predicted=np.mean(y_predicted_list)
                xx_list.append(mean_predicted)
            xxx_list.append([land,xx,xx_list])
            print([land,xx,xx_list])
               
with open('xxx_list_r.py', 'w') as w:
    w.write('xxx_list_r = %s' % xxx_list)       