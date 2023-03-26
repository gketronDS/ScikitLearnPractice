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
plt.hist((x - np.mean(x))/np.std(x),30)
plt.show()