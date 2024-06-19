Big_list=[]
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
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


cat_nr=100
import pandas as pd
pd.set_option('display.max_columns', 500)
month_data_encoded=pd.read_excel(r"month_data_encoded.xlsx")
mde=month_data_encoded
mde=mde.drop(['dateRep', 'cases' , 'Rescd' , '7days_before_mean','7days_after_mean', 'week' ,'year' , 'index', 'month', 'countriesAndTerritories', 'va' , 'vaccin'], axis=1)
mde = mde.reindex(sorted(mde.columns), axis=1)
percent_list=[]
number_list=[-10000]
for percent,number in (enumerate(mde['rp_zeitraum'].describe([x/cat_nr for x in range (1,cat_nr)])[4:cat_nr+3].tolist())):
    percent_list.append(int(percent))
    number_list.append(number)
    
percent_list.append(int(cat_nr-1))    
number_list.append(10000)    
mde['rp_zeitraum'] = pd.cut(x=mde['rp_zeitraum'], bins=number_list,labels=percent_list)


train, test = train_test_split(mde, test_size=0.25)
#test=test.reset_index()
Y_test = test["rp_zeitraum"]
X_test = test.drop(labels = ["rp_zeitraum"],axis = 1)
X_test = X_test.values.reshape(-1,1,111,1)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 100)
ac_list=[]
n_mc_run = 1
for i in range(0, 100):
    y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
    y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
    ac=accuracy_score([np.argmax(Y_test[idx], axis=-1) for idx in range (0,len(Y_test))],[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in      range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(Y_test)) ])
#accuracy(model, labels)
    ac_list.append(ac)
Big_list.append(["Base line",ac_list])
print("Base line")

test_feature_shuffled=test
test_feature_shuffled_part_month=test_feature_shuffled[['month_0','month_1','month_2','month_3']]
test_feature_shuffled_minus_month=test_feature_shuffled.drop([ 'month_0','month_1','month_2','month_3'], axis=1)
test_feature_shuffled=pd.concat([test_feature_shuffled_minus_month, test_feature_shuffled_part_month.sample(frac = 1)], axis=1)#CBook_i['dateRep'].dt.week
#test_feature_shuffled[xx] = np.random.permutation(test_feature_shuffled[xx].values)
#test_feature_shuffled=  test.assign(**{xx:np.random.permutation(test[xx])})#
Y_test = test_feature_shuffled["rp_zeitraum"]
X_test = test_feature_shuffled.drop(labels = ["rp_zeitraum"],axis = 1)
X_test = X_test.values.reshape(-1,1,111,1)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 100)
ac_list=[]
n_mc_run = 1
for i in range(0, 100):
    y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
    y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
    ac=accuracy_score([np.argmax(Y_test[idx], axis=-1) for idx in range (0,len(Y_test))],[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in      range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(Y_test)) ])
#accuracy(model, labels)
    ac_list.append(ac)
Big_list.append(["Month",ac_list])
print("Month")
#de.drop(['countriesAndTerritories_0', 'countriesAndTerritories_1', 'countriesAndTerritories_2', 'countriesAndTerritories_3','countriesAndTerritories_4'
test_feature_shuffled=test
test_feature_shuffled_part_month=test_feature_shuffled[['countriesAndTerritories_0', 'countriesAndTerritories_1', 'countriesAndTerritories_2', 'countriesAndTerritories_3','countriesAndTerritories_4']]
test_feature_shuffled_minus_month=test_feature_shuffled.drop(['countriesAndTerritories_0', 'countriesAndTerritories_1', 'countriesAndTerritories_2', 'countriesAndTerritories_3','countriesAndTerritories_4'], axis=1)
test_feature_shuffled=pd.concat([test_feature_shuffled_minus_month, test_feature_shuffled_part_month.sample(frac = 1)], axis=1)#CBook_i['dateRep'].dt.week
#test_feature_shuffled[xx] = np.random.permutation(test_feature_shuffled[xx].values)
#test_feature_shuffled=  test.assign(**{xx:np.random.permutation(test[xx])})#
Y_test = test_feature_shuffled["rp_zeitraum"]
X_test = test_feature_shuffled.drop(labels = ["rp_zeitraum"],axis = 1)
X_test = X_test.values.reshape(-1,1,111,1)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 100)
ac_list=[]
n_mc_run = 1
for i in range(0, 100):
    y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
    y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
    ac=accuracy_score([np.argmax(Y_test[idx], axis=-1) for idx in range (0,len(Y_test))],[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in      range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(Y_test)) ])
#accuracy(model, labels)
    ac_list.append(ac)
Big_list.append(["Country",ac_list])
print("Country")
mde_1=mde.drop(['countriesAndTerritories_0', 'countriesAndTerritories_1', 'countriesAndTerritories_2', 'countriesAndTerritories_3','countriesAndTerritories_4', 'month_0','month_1','month_2','month_3','rp_zeitraum'], axis=1)
for xx in mde_1.columns.tolist():
    test_feature_shuffled=test
    #test_feature_shuffled[xx] = np.random.permutation(test_feature_shuffled[xx].values)
    test_feature_shuffled=  test.assign(**{xx:np.random.permutation(test[xx])})#
    Y_test = test_feature_shuffled["rp_zeitraum"]
    X_test = test_feature_shuffled.drop(labels = ["rp_zeitraum"],axis = 1)
    X_test = X_test.values.reshape(-1,1,111,1)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes = 100)
    ac_list=[]
    n_mc_run = 1
    for i in range(0, 100):
        y_pred_logits_list = [model(X_test) for _ in range(n_mc_run)]  # a list of predicted logits
        y_pred_prob_all = np.concatenate([tf.nn.softmax(y, axis=-1)[:, :, np.newaxis] for y in y_pred_logits_list], axis=-1)
        ac=accuracy_score([np.argmax(Y_test[idx], axis=-1) for idx in range (0,len(Y_test))],[np.argmax([np.mean(y_pred_prob_all[idx][i])  for i in      range (0,len(y_pred_prob_all[idx]))],axis=-1) for idx in range (0,len(Y_test)) ])
    #accuracy(model, labels)
        ac_list.append(ac)
    Big_list.append([xx,ac_list])
    print(xx)
with open('Big_list.py', 'w') as w:
    w.write('Big_list = %s' % Big_list)