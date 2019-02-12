# -*- coding: utf-8 -*-
"""
Created on Sat May  5 19:11:51 2018

@author: brahe
"""

# =============================================================================
# PlEASE MAKE SURE THAT TENSOR FLOW AND KERAS 2.1.3 IS INSTALLED 
# IF NOT USE THE COMMAND: pip3 install --upgrade keras==2.1.3
# We are using: Python 3.5.4 |Anaconda, Inc.| (default, Nov  8 2017, 14:34:30) [MSC v.1900 64 bit (AMD64)]
# IPython 6.2.1 -- An enhanced Interactive Python.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# Importing the Keras libraries 

from keras.layers import Dropout
from keras.layers.core import  Activation
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.models import model_from_json

#Set the working directory to the folder data 

# =============================================================================
#                 Part 1: Building model      
# =============================================================================


# Importing the dataset
def read_data():
    y = pd.read_csv('Classes1000.csv' , sep = ';')
    X = pd.read_csv('portf1000.csv' , sep = ';')
    X = X.iloc[:, 1:].values
    y = y.iloc[:, 2].values
    y = y-1
    return X, y




# Tranforming y to the same length of X
def data_same_length(X, y, nb_of_portfolios, size_of_portfolio):     
    y_portfolios = []
    for i in range(nb_of_portfolios):
        for j in range(size_of_portfolio):
            y_portfolios.append(y[i,])
    return y_portfolios


 
# Split data into train and test       
def split_train_test(grou, train_size,
                     portfolio_size,
                     nb_of_features,
                     nb_of_portfolios):
    
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    
    for k in range(nb_of_portfolios):
        if k < int(train_size * nb_of_portfolios):
            X_train.append(grou[k, 0:portfolio_size, 0:nb_of_features])
            y_train.append(grou[k, 0:1, nb_of_features:nb_of_features+1])
        else:
            X_test.append(grou[k, 0:portfolio_size, 0:nb_of_features])
            y_test.append(grou[k, 0:1, nb_of_features:nb_of_features+1])
            
    X_test = np.array(X_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_test, X_train, y_train, y_test


# Encoding categorical data
def transform_categorical_variables(data):
    labelencoder_X_1 = LabelEncoder()
    data = labelencoder_X_1.fit_transform(data)
    labelencoder_X_2 = LabelEncoder()
    
    data = data.reshape(len(data),1)
    onehotencoder = OneHotEncoder(categorical_features = [0])
    data = onehotencoder.fit_transform(data).toarray()
    return data






# Convolutional NN architecture

def convolutional_NN_model(X_train, X_test, y_train,
                           y_test,
                           batch_size,
                           nb_epoch):
    
    # Step1 Convolutional layer
    classifier = Sequential()
    classifier.add(Conv2D(32, (1, 3), input_shape = (1, 20, 38),
                          activation = 'tanh'))
    
    # Step 2 - Pooling
    
    classifier.add(MaxPooling2D(pool_size = (1, 3)))
    
    # Step 2(b) - Add 2nd Convolution Layer making it Deep followed by a Pooling Layer
    
    classifier.add(Conv2D(32, (1, 3), activation = 'tanh'))
    classifier.add(MaxPooling2D(pool_size = (1, 2)))
    
    # Step 3 - Flattening
    
    classifier.add(Flatten())
    
    # Step 4 - Fully Connected Neural Network
    
    # Hidden Layer - Activation Function RELU
    classifier.add(Dense(units = 256, activation = 'tanh')) 
    # Output Layer - Activation Function Softmax(to clasify classes)
    classifier.add(Dense(units = 2, activation = 'softmax'))
    
    # Compile the CNN
    
    # Binary Crossentropy - to classify between good and bad portfolios
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    
    classifier.fit(X_train, y_train,batch_size, nb_epoch,
                   validation_data=(X_test, y_test))
    
    return classifier


# transforming 2d array into a list

def transform_matrix_tolist(y_test):
    y_test_array = []
    for f in range(len(y_test)):
        if y_test[f][0] == 1:
            y_test_array.append(0) 
        elif y_test[f][1] == 1:
            y_test_array.append(1)
            
    y_test_array = np.array(y_test_array) 
    return y_test_array






if __name__ == "__main__":
    
    # Importing the dataset
    X, y = read_data()
    
    # Tranforming y to the same length of X
    y_portfolios = data_same_length(X, y, nb_of_portfolios = 1000,
                                    size_of_portfolio = 20)
     
    # missing data 
    X = pd.DataFrame(X)    
    X = X.fillna(0)
    
    # Concatenate X and y
    y = pd.DataFrame(y)
    y = np.array(y)
    y_portfolios = np.array(y_portfolios) 
    y_portfolios = pd.DataFrame(y_portfolios) 
    grou = pd.concat((X,y_portfolios),axis=1)
    grou = np.array(grou)
    
    # Reshape grou into 3 dimensions tensor
    grou = grou.reshape(y.shape[0], 20, 39)
  
    # Split data into train and test
    X_test, X_train, y_train, y_test = split_train_test(grou,
                                                        train_size = 0.8,
                                                        portfolio_size = 20,
                                                        nb_of_features = 38,
                                                        nb_of_portfolios = 1000)
    


    # Reshape y
    y_test = y_test.reshape(200,1)
    y_train = y_train.reshape(800,1)

    # Encoding categorical data
    y_train = transform_categorical_variables(data = y_train)
    y_test = transform_categorical_variables(y_test)
    
    # Reshape into 4d tensors
    X_train = np.reshape(X_train, (800, 1, 20, 38))
    X_test = np.reshape(X_test, (200, 1, 20, 38))
    
        #fit model
    classifier = convolutional_NN_model(X_train, X_test, y_train, y_test, 1, 100)
    
    
# =============================================================================
#                     Part 2: Model evaluation
# =============================================================================
    
    # Prediction
    y_pred = classifier.predict_classes(X_test, batch_size = 1)
    
    # Transforming 2d array into a list
    y_test_array = transform_matrix_tolist(y_test) 
        # Confusion matrix
    confusion_matrix(y_pred, y_test_array)
    
    # Classification report
    from sklearn.metrics import classification_report
    target_names = ['good', 'bad']
    print(classification_report(y_pred, y_test_array, target_names=target_names))

    
    
    
# =============================================================================
#                   Part 3:   Model summary
# =============================================================================

# Model summary (architectures)
classifier.summary()

# model Configurations
classifier.get_config()

# Number of parameters in the model
classifier.count_params()

# Model weights
classifier.get_weights()

      
# =============================================================================
#              Part 4:  Save trained model parameters         
# =============================================================================

## serialize weights to HDF5
#classifier.save_weights("pre_trained/final_model.h5")

#print("Saved model to disk")
#model_json = classifier.to_json()
#with open("pre_trained/final_model", "w") as json_file:
#    json_file.write(model_json)

# =============================================================================
#                 Part5: Load Pre-trained model
# =============================================================================
 
# load json and create model
json_file = open('pre_trained/final_model', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("pre_trained/final_model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

loaded_model.fit(X_train, y_train,batch_size=1, nb_epoch=1,
                 validation_data=(X_test, y_test))

# evaluate loaded model on test data
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print(score)



# =============================================================================
#                  Part 6:Confusion matrix
# =============================================================================

       




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_array, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ("good", "bad"),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=("good" "bad"), normalize=True,
                      title='Normalized confusion matrix')

plt.show()



