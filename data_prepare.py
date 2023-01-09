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
    X_train, X_valid, y_train, y_valid =  train_test_split(X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED)  

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def prepare_infer_data(attribute_values):

    if attribute_values[0] == 'Female':
        attribute_values[0] = 1
    else:
        attribute_values[0] = 0

    if attribute_values[1] == 'Loyal Customer':
        attribute_values[1] = 1
    else:
        attribute_values[1] = 0

    if attribute_values[3] == 'Business travel':
        attribute_values[3] = 2
    else:
        attribute_values[3] = 1
    
    if attribute_values[4] == 'Business':
        attribute_values[4] = 3
    elif attribute_values[4] == 'Eco Plus':
        attribute_values[4] = 2
    else:
        attribute_values[4] = 1
    
    for i in range(6, 20):
        attribute_values[i] = int(attribute_values[i])

    return attribute_values