import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pandas as pd

warnings.filterwarnings('ignore')

dataFrame = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv')
data = dataFrame.to_numpy()
X, y = data[:, :-1], data[:, -1]

# Split data into train partition and test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.25)

mlp = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=20,
    alpha=1e-4,
    solver="adam",
    random_state=1,
    activation="tanh",
)

# this example won't converge because of resource usage constraints on
# our Continuous Integration infrastructure, so we catch the warning and
# ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

y_preds = mlp.predict(X_test)
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, mlp.predict(X_train)))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_preds))
print("#" * 40 + "\n")

cm = confusion_matrix(y_test, y_preds, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
cmd.plot()
