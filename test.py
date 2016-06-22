import pandas
df1 = pandas.read_csv('wine.txt')
Y = df1['1'].values
X = df1[ ['2','3','4','5','6','7','8','9','10','11','12','13','14'] ].as_matrix()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import svm
from sklearn import neighbors
kf = KFold(Y.size, n_folds=5, shuffle=True, random_state=42)
a=[]
i=1
while i<51:
    knn = neighbors.KNeighborsClassifier(n_neighbors = i)
    scores = cross_validation.cross_val_score(knn, X, Y, cv = kf)
    a.append(scores.mean())
    i=i+1
print(a.index(max(a))+1)
