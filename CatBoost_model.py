import data_prepare as dp
from catboost import CatBoostClassifier 
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# get data
path = 'data/Invistico_Airline.csv'
X_train, y_train, X_valid, y_valid, X_test, y_test = dp.data_prepare(path)

# model
CatBoost_model = CatBoostClassifier(
    iterations=50,
    random_seed=63,
    learning_rate=0.1,
    custom_loss=['Accuracy'],
    eval_metric='Accuracy' 
)
# train
CatBoost_model.fit(
    X_train, y_train,
    cat_features=dp.features,
    eval_set=(X_valid, y_valid),
    logging_level='Silent',
    plot=False
)
# test
y_pred = CatBoost_model.predict(X_test)
# result
print(classification_report(y_pred,y_test))
print('accuracy_score',accuracy_score(y_pred, y_test))