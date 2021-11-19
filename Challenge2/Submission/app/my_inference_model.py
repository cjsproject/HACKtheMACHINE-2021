# THIS IS AN EXAMPLE SCRIPT. YOU CAN USE THIS DIRECTLY OR ADAPT IT TO YOUR NEEDS.

'''
Hackthemacine 2021 Model Inference Script

This is an example of how you might want to organize your model so it can be
easily called by a testing script (such is the case for "my_testing_script.py").

You could also just put all this code into your testing script file.
It is not strictly necessary to split the model into a separate file like this.

Feel free to use this script directly or adapt it as needed for your solution.

What this script does:
1) Load your model
2) Make a prediction on a single piece of data
'''


import pickle
import random
import sklearn
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import pandas as pd
from copy import deepcopy

class MyClassifier:
    
    def __init__(self):
        # good place to load my awesome model here
        self.clf = pickle.load(open('pipeline.pickle', 'rb'))


    def preproc(self, data):
        # take the numeric values from the df
        data = deepcopy(data)
        xt = np.array(data.select_dtypes(include=np.number).values)

        # save just the objects(non-numeric), drop the sha256 column
        tst = data.select_dtypes(include='object').drop('sha256', axis=1)

        # function returns a key(and int between 0 and maximum unique values for the column) for each value input
        def find_key(input_dict, value):
            return next((k for k, v in input_dict.items() if v == value), None)

        # populates a new vector with enumerated values of unique objects
        uniques = dict(enumerate(data.section_entry.unique()))

        # create empty dataframe for containing objects
        us = np.empty([xt.shape[0], 1])

        # populate column vector with enumerated values
        for i in range(xt.shape[0]):
            us[i] = find_key(uniques, data.section_entry[i])

        us = np.nan_to_num(us, nan=len(uniques)+1)

        # concatenate the objects and the numeric columns
        xt = np.hstack((xt, us))

        return xt

    def predict(self, data):
        # good place to do whatever data transformations we need - this is done in preproc()

        # good place to have a model.predict() kind of call

        # run prediction on sklearn pipeline.
        prediction = self.clf.predict(data)

        return(prediction)
