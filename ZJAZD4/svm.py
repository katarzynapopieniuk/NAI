import numpy as np
import warnings
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm

warnings.filterwarnings('ignore')
def teach_svm(sourceFile) :
        data = np.loadtxt(sourceFile, delimiter=',')
        X, y = data[:, :-1], data[:, -1]

        # Split data into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=5)

        svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

        #Train the model using the training sets
        svc.fit(X_train, y_train)

        #Predict the response for test dataset
        y_pred = svc.predict(X_test)

        # Evaluate classifier performance
        print("\nSVM")
        print("\n" + "#"*40)
        print("\nClassifier performance on training dataset\n")
        print(classification_report(y_train, svc.predict(X_train)))
        print("#"*40 + "\n")

        print("#"*40)
        print("\nClassifier performance on test dataset\n")
        print(classification_report(y_test, y_pred))
        print("#"*40 + "\n")
