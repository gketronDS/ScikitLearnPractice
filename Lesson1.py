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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_california_housing(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pylab as plt

model = KNeighborsRegressor().fit(X, y)
pred = model.predict(X)
plt.scatter(pred, y)
plt.show()

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#image = mpimg.imread('1600px-Exit_Ramp.jpeg')
#plt.imshow(image)
#plt.show()
