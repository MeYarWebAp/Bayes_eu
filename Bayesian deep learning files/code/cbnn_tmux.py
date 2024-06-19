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

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import time
from sklearn.model_selection import train_test_split

#%matplotlib inline
#random seed as the birthday of my granp which is in the hospital fighting with cancer
#be strong Valdomiro!
#np.random.seed(10171927)
#tf.random.set_seed(10171927)
from sklearn.model_selection import train_test_split
train, test = train_test_split(mde, test_size=0.10)
Y_train = train["rp_zeitraum"]
# Drop 'label' column
X_train = train.drop(labels = ["rp_zeitraum"],axis = 1)
X_train = X_train.values.reshape(-1,1,111,1)
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 100)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state=42)
Y_t = test["rp_zeitraum"]
# Drop 'label' column
X_t = test.drop(labels = ["rp_zeitraum"],axis = 1)
X_t = X_t.values.reshape(-1,1,111,1)
Y_t = tf.keras.utils.to_categorical(Y_t, num_classes = 100)

#ypred_test=X_t.tolist()

with open('X_t.py', 'w') as w:
    w.write('X_t = %s' % X_t.tolist())
with open('Y_t.py', 'w') as w:
    w.write('Y_t = %s' % Y_t.tolist())
with open('X_train.py', 'w') as w:
    w.write('X_train = %s' % X_train.tolist())
with open('Y_train.py', 'w') as w:
    w.write('Y_train = %s' % Y_train.tolist())
with open('X_val.py', 'w') as w:
    w.write('X_val = %s' % X_val.tolist())
with open('Y_val.py', 'w') as w:
    w.write('Y_val = %s' % Y_val.tolist())

def build_bayesian_bcnn_model(input_shape):
    
    """
    Here we use tf.keras.Model to use our graph as a Neural Network:
    We select our input node as the net input, and the last node as our output (predict node).
    Note that our model won't be compiled, as we are usign TF2.0 and will optimize it with
    a custom @tf.function for loss and a @tf.function for train_step
    Our input parameter is just the input shape, a tuple, for the input layer
    """
    
    model_in = tf.keras.layers.Input(shape=input_shape)
    conv_1 = tfp.python.layers.Convolution2DFlipout(32, kernel_size=(1, 3), padding="same", strides=1)
    x = conv_1(model_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    conv_2 = tfp.python.layers.Convolution2DFlipout(64, kernel_size=(1, 3), padding="same", strides=1)
    x = conv_2(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    dense_1 = tfp.python.layers.DenseFlipout(512, activation='relu')
    x = dense_1(x)
    dense_2 = tfp.python.layers.DenseFlipout(100, activation=None)
    model_out = dense_2(x)  # logits
    model = tf.keras.Model(model_in, model_out)
    return model

@tf.function
def elbo_loss(labels, logits):
    loss_en = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    loss_kl = tf.keras.losses.KLD(labels, logits)
    loss = tf.reduce_mean(tf.add(loss_en, loss_kl))
    return [loss, tf.reduce_mean(loss_en), tf.reduce_mean(loss_kl)]

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = bcnn(X_train)
        loss = elbo_loss(labels, logits)[0]
    gradients = tape.gradient(loss, bcnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bcnn.trainable_variables))
    return loss

def accuracy(preds, labels):
    #return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))
    predictednumbers=np.argmax(preds, axis=1)
    realnumbers=np.argmax(labels, axis=1)
    diff=np.subtract(predictednumbers , realnumbers)
    return np.sqrt(sum(np.power(diff, 2))/len(diff))

bcnn = build_bayesian_bcnn_model(X_train.shape[1:])
optimizer = tf.keras.optimizers.Adam(lr=0.001)

times = []
accs = []
val_accs = []
losses = []
val_losses = []
val_losses_en = []
val_losses_kl = []
for i in range(1000):
    #tic = time.time()
    loss = train_step(X_train, Y_train).numpy()
    preds = bcnn(X_train)
    acc = accuracy(preds, Y_train)
    accs.append(acc)
    losses.append(loss)
    
    val_preds = bcnn(X_val)
    val_loss = elbo_loss(Y_val, val_preds)[0].numpy()
    val_loss_en=elbo_loss(Y_val, val_preds)[1].numpy()
    val_loss_kl=elbo_loss(Y_val, val_preds)[2].numpy()
    val_acc = accuracy(Y_val, val_preds)
    
    val_accs.append(val_acc)
    val_losses.append(val_loss)
    val_losses_en.append(val_loss_en)
    val_losses_kl.append(val_loss_kl)
    #tac = time.time()
    #train_time = tac-tic
    #times.append(train_time)
    
    print("Epoch: {}: loss = {:7.3f} , accuracy = {:7.3f}, val_loss = {:7.3f}, val_acc={:7.3f} ".format(i, loss, acc, val_loss, val_acc))
bcnn.save("convmodel")
with open('accs.py', 'w') as w:
    w.write('accs = %s' % accs)
with open('losses.py', 'w') as w:
    w.write('losses = %s' % losses)
with open('val_accs.py', 'w') as w:
    w.write('val_accs = %s' % val_accs)
with open('val_losses.py', 'w') as w:
    w.write('val_losses = %s' % val_losses)
with open('val_losses_en.py', 'w') as w:
    w.write('val_losses_en = %s' % val_losses_en)
with open('val_losses_kl.py', 'w') as w:
    w.write('val_losses_kl = %s' % val_losses_kl)