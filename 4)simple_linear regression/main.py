import numpy as np

class MyLR:

    def __init__(self):
        self.m = None
        self.b = None

    def fit(self,X_train,y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num += ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den += ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))

        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self,X_test):
        print(X_test)
        return self.m * X_test + self.b


df = pd.read_csv('placement.csv')
# extracting X and y
X = df.iloc[:,0].values
y = df.iloc[:,1].values
#training the model
#splitting data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
#lr will have fit and transform
lr = MyLR()
#fit the model with training data
lr.fit(X_train,y_train)
#gives slope(m) and b(constant)
print(lr.predict(X_test[0]))
#gives test value first -> given out by the function
#predicts for the test value
