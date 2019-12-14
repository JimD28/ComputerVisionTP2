import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Getting back the objects:
with open('dataset_resized.pkl', 'rb') as g:
    ellipses, labels = pickle.load(g)

ellipses = ellipses.swapaxes(0,2)
ellipses2 = ellipses.reshape(5369,76800)

X_train, X_test, y_train, y_test = train_test_split(ellipses2, labels, test_size=0.999, random_state = 42)

loaded_model = pickle.load(open('GraDB60020.8True93.97%.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
y_predict = loaded_model.predict(X_test)
print(result)
print(confusion_matrix(y_test, y_predict))
print("\n \n")
target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(y_test, y_predict, target_names=target_names))
