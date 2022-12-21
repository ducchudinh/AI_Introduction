import pandas as pd 
from sklearn.model_selection import train_test_split

features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Seat comfort',
               'Departure/Arrival time convenient', 'Food and drink', 'Gate location',
               'Inflight wifi service', 'Inflight entertainment', 'Online support',
               'Ease of Online booking', 'On-board service', 'Leg room service',
               'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']
target = 'satisfaction'

def data_prepare(path):
    """prepare data for train, validate and test process

    Args:
        path (string): path to .csv file

    Returns:
        _type_: _description_
    """
    airlineDF = pd.read_csv(path)

    # drop rows containing nan value
    airlineDF = airlineDF.dropna()

    # satisfaction - label encoding
    satisfaction_mapping = {'satisfied': 1, 'dissatisfied': 0}
    airlineDF['satisfaction'] = airlineDF['satisfaction'].map(satisfaction_mapping)
    # Gender - encoding
    gender_mapping = {'Female': 1, 'Male': 0}
    airlineDF['Gender'] = airlineDF['Gender'].map(gender_mapping)
    # Customer Type - encoding
    customer_type_mapping = {'Loyal Customer': 1, 'disloyal Customer': 0}
    airlineDF['Customer Type'] = airlineDF['Customer Type'].map(customer_type_mapping)
    # Type of Travel - encoding
    travel_type_mapping = {'Business travel': 2, 'Personal Travel': 1}
    airlineDF['Type of Travel'] = airlineDF['Type of Travel'].map(travel_type_mapping)
    # Class - encoding
    class_mapping = {'Business': 3, 'Eco Plus': 2, 'Eco': 1}
    airlineDF['Class'] = airlineDF['Class'].map(class_mapping)

    # split data
    X = airlineDF.drop([target], axis=1) 
    y = airlineDF[target]
    SEED = 42 
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED) 
    VAL_SIZE = 0.2
    X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED)  

    return X_train, y_train, X_val, y_val, X_test, y_test