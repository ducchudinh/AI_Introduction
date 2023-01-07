import data_prepare as dp
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# get data
path = 'data/Invistico_Airline.csv'
X_train, y_train, X_valid, y_valid, X_test, y_test = dp.data_prepare(path)

# model
LinearSVC_model = make_pipeline(StandardScaler(), LinearSVC())

# train
LinearSVC_model.fit(X_train, y_train)

# test
y_pred = LinearSVC_model.predict(X_test)

# result
print(classification_report(y_pred,y_test))
print('accuracy_score',accuracy_score(y_pred, y_test))

# save model
import pickle
pickle.dump(LinearSVC_model, open('LinearSVC_trained_model', 'wb'))