# Lesson 1: Introduction.
# python3.10 -m venv studysession
#source studysession/bin/activate
#pip install --upgrade pip
#pip install --upgrade matplotlib
#pip install --upgrade scikit-learn==0.23.0 (Failed - missing numpy)
#pip install --upgrade numpy
#pip install --upgrade scikit-learn
#pip install --upgrade pandas
#pip install --upgrade seaborn

#how ML works: data => model => prediction
#1. Split data into X and Y sections
#2. X = data to build prediction on, Y = set to test prediction against.
#Practice use case is house price prediction. X = info about house Y = prices
#3. Pass to model to build prediction.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print(fetch_california_housing()['DESCR'])
X, y = fetch_california_housing(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pylab as plt

model = KNeighborsRegressor().fit(X, y)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
])
pipe.get_params()
#Turn Pipeline into a grid search w paramgrid be 
mod = GridSearchCV(estimator=pipe,
                   param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                   cv=3)

mod.fit(X, y)
cvresult = pd.DataFrame(mod.cv_results_)
print(cvresult)
#Dot looks for its nearest neighbors w distance.
# Prediction is average of 5 nearest neighbors. 
# What if the number scales are different? 
# That axis has a much bigger effect. 
# Rethink model, with preprocessing on the X before it gets to the model
# scale and knearestneighbor is the new model. Pipeline gets considered the model in,
# can call fit and predict off of pre processing.   
#After scaling, need to change possible settings to optimize settings
#you also want to compare label and prediciton. 
#Also cut data X into 3 segments, and copy x and y 3 times. 
#Pipeline cant handle this optimization, use a grid search CV object. 
#used for cross validation. 
pred = mod.predict(X)
plt.scatter(pred, y)
plt.show()
#Not ready to go to production...
#Need to do differently to understand what is happening in the dataset.
#DO NOT TRUST YOUR MODEL UNTIL YOU TRY TO EXPLOIT IT
#Need to stress test your model predictions
#issue with grid search

#Hard part of DS is understanding what data is saying and what to do when
#model is in production. Know ethics and algos (Feedback loops, and fallback scenarios)


#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#image = mpimg.imread('1600px-Exit_Ramp.jpeg')
#plt.imshow(image)
#plt.show()
