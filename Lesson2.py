#Preprocessing Techniques with Scikit Learn
#What frequent transformers do people use?
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
df = pd.read_csv("drawndata1.csv")
print(df.head(3))

X = df[['x','y']].values
y = df['z'] == "a"
#plt.scatter(X[:,0], X[:,1],c=y)
#plt.show()

#We want to rescale the axes w Standard Scalar. 
#each column has a mean and var calculated.
# Reading [x-mean(x)]/sqrt(var) normalizes reading around 0

from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
#plt.scatter(X_new[:,0], X_new[:,1],c=y)
#plt.show()

#Y spread is 8 units
#X spread is 3 and 1/2. 

#Only proportionally smaller...
#Example of what Standard Scalar is doing wrong: 

x = np.random.exponential(10,(1000)) + np.random.normal(0,1,(1000))
#plt.hist((x - np.mean(x))/np.std(x),30)
#plt.show()

#still has outliers... 
#how to normalized to make outliers less of a problem.
#could base off quantiles.. median, 25 quant, 75 quant...
#how to project onto something normalized.
#Can map onto quant location numberline.
#Use the quantile transformer.

from sklearn.preprocessing import QuantileTransformer
X_new_quant = QuantileTransformer(n_quantiles=252).fit_transform(X)
#plt.scatter(X_new_quant[:,0], X_new_quant[:,1],c=y)
#plt.show()

#Quantile wants 1000 but we dont have that many samples, 
# so we set quantiles to the number available: 252
#outliers are less impactful. 
#Now we pass this quantile scale to get better predictions. 
#plot_output(scaler= StandardScaler())
#plot_output(scaler= QuantileTransformer(n_quantiles=252))
#then verify with the gridsearchCV object.

#now to try regressions

df = pd.read_csv("drawndata2.csv")
print(df.head(3))

X = df[['x','y']].values
y = df['z'] == "a"
#plt.scatter(X[:,0], X[:,1],c=y)
#plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", PolynomialFeatures()),
    #("scale", QuantileTransformer(n_quantiles=100)),
    ("model", LogisticRegression())
])
pred = pipe.fit(X,y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.show()

#need to get a seperating line with X1*X2 or X1^2.
#Need a nonlinear regression to deliniate groups.


arr = np.array(["low","low","high","medium"]).reshape(-1,1)
print(arr)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
print(enc.fit_transform(arr))
print(enc.transform([["zero"]]))

#can encode categories