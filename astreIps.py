import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans


from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler,SMOTE


data=pd.read_csv("Questionnaire_3A-1.csv")

data2=data.drop(columns=['Timestamp']);

obj_data = data2.select_dtypes(include=['object']).copy()

data2_dict = obj_data.to_dict(orient='records') # turn each row as key-value pairs

# DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# instantiate a Dictvectorizer object for X
data2_dv = DictVectorizer(sparse=False)
# sparse = False makes the output is not a sparse matrix

# apply dv_X on X_dict
data_encoded = data2_dv.fit_transform(data2_dict)

new = pd.DataFrame.from_dict(data_encoded)

dataf=pd.concat([data2["1-Quel est ton numéro étudiant ?"],pd.DataFrame(new,index=data2.index),] ,axis=1)

kmeans = KMeans(2)
kmeans.fit(new)
pred = kmeans.predict(new)
pred2=pred.reshape(-1,1)

df = pd.DataFrame(data=pred2,columns=["spécialité"])

datafinale=pd.concat([df, new], axis=1)

X= datafinale.iloc[:, 1:307].values
y = datafinale.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 , random_state=42)

from imblearn.metrics import classification_report_imbalanced


rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train, y_train)
# Entraînement du modèle de régression logistique
lr = LogisticRegression()
lr.fit(X_ro, y_ro)
# Affichage des résultats
y_pred = lr.predict(X_test)


knn = KNeighborsClassifier(7)
knn_model = knn.fit(X_ro, y_ro)
y_pred_knn =knn_model.predict(X_test)

# Fit the model on training set
#model = LogisticRegression()
#model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(knn_model, open(filename, 'wb'))


model=pickle.load(open(filename,'rb'))
