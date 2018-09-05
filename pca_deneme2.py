import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

names = ['1', '2', '3', '4', '5', '6', '7', '8','9','10',
         '11', '12', '13', '14', '15', '16', '17', '18','19','20',
         '21', '22', '23', '24', '25', '26', '27', '28','29','30',
         '31', '32', '33', '34', '35', '36', '37', '38','39','40',
         '41', '42', '43', '44', '45', '46', '47', '48','49','50',
         '51', '52', '53', '54', '55', '56', '57', '58','59','60','class']
#df = pd.read_excel("C:/Users/ASUS/Desktop/ciktilar/nan_class_ayri_siniflar/class_1(_2_3_)4 min max normalize.xlsx")

df = pd.read_excel("C:/Users/ASUS/Desktop/ciktilar/nan_class_ayri_siniflar/ayri_yeni_normalizasyon/nan_1_2_ min_max.xlsx")
array = df.values
X = array[:,0:60]# train data value
y = array[:,60]
#X = iris.data
#y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
for i in range(1,60,3):
    pca = PCA(n_components=i)# adjust yourself
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    clf = SVC()
    clf.fit(X_t_train, y_train)
    #print 'score',clf.score(X_t_test, y_test),'for components number = ',i
    predicted = clf.predict(X_t_test)
    # get the accuracy
    print'for components number = ',i, accuracy_score(y_test, predicted)


    #print 'accurancy',clf.accuracy_score(X_t_test, y_test),'for components number = ',i

#print 'pred label', clf.predict(X_t_test)