from decision_tree import teach_decision_tree
from svm import teach_svm

print("\nTraining on Pima Indians Diabetes Dataset")
teach_decision_tree('indian-diabetes.csv')
teach_svm('indian-diabetes.csv')
print("\nTraining on Milk Quality Dataset")
teach_decision_tree('milknew.csv')
teach_svm('milknew.csv')
