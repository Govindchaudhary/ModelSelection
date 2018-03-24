'''
<-----------Improving the model performance--------------------->
basically there are 2 types of parameters :
    1) that learn from our machine learning models
    2) the parameters that we have to choose also called as hyperparameters
so we can improve our model's performance by choosing the optimal values of hyperparameters

we are using grid_search technique to know how to find the optimal value of hyperparameters
of a particular model

Also,
<-------now an obvious question is which model to choose to solve a particular problem
or we can say which moddel is the best for solving our problem?????

--------> to answer this question you need to find two things:
    1)the type of your problem:
    can be easily find out by looking at the dependent variable of your dataset
    if:
        1)dependent variable is not present :then problem is clustering one
        2)if DV is caterogical variable then it is classification problem
        3) and last if DV contains continuous real values then regression problem
    
    2)your dataset is linearly seperable or not:
       well this is not easy to find by just looking at the datset
       for this we will use grid_search technique
    
    so we are working on the same social network clssification problem:
        firstly we find weather to choose linear model like svm
        or non-linear model like kernel_svm
        then choosing the optimal values of hypeerparameters to improve the model
        performance.
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#apply the k-fold cross validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,y = y_train , cv=10)
accuracies.mean()
accuracies.std()

#applying the grid_search methd to find the best model(for eg.  linear or non-linear) and best value of hyperparameters.
from sklearn.model_selection import GridSearchCV
#the hyperameters that we have to optimze which is a list of dictionaries
'''
firstly it find which kernel is best suited for this problem:
    if kernel is linear it means data is linearly sperable
    so then we will find the optimal value of penalty parameter C which is to reduce the chance of overfitting
    more the value of C less the chance of overfitting
    but also remember if it is very large the their might occur situation of underfitting
then it will check for non-linear kernel ie. rbf
and once it finds thst it is best suited for this then it will find:
    optimal value of C and also gamma(more the no. of your featues less value of gamma can be optimal for your problem)

''' 
parameters = [{'C':[0,10,100,1000],'kernel':['linear']},
              {'C':[0,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,.001,.0001]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy', # criterion to optimze the value of hyperparameters ie. we want to optimize our model on the base of accuracy or precision or some other performance criterion
                           cv = 10, # we are using 10-fold cross validatio to evaluate the model performance
                           n_jobs=-1) #utilizing all the power of your cpu

grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_   # best accuracy obtained from best value of hyperparameters
best_params = grid_search.best_params_   # best optimal values of hyper parameters


# Visualising the Training set results(useful for non-linear data)
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()