#imports 
path = '../'
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

## These functions are intended to predict potential resources from a variety of online sources e.g. twitter, github, reddit etc... 

def lr_training(t_input, t_feature, target, cv_int, max_feats, filename, path):
    """Create a text classifier model based on Logistic Regression and TF-IDF. Use cross validation

    Parameters
    ----------
    t_input:
        training set input, dataframe (var)
    t_feature:
        df column title, values should be string formatted (str)
    target:
        df column, [0,1] values (int)
    cv_int: 
        the number of cross validation folding (int)
    max_feats:  
        set the feature number for tdidf transformer (int)
    filename: 
        model file name to be saved (str)
    path: 
        parent folder (str/var)
    """
    
    #pipeline setup w/ transformer, algorithm and training inputs
    tfidf_transformer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=max_feats)
    lr = LogisticRegression(solver='liblinear', C=10.0, random_state=44)

    x_train = t_input[t_feature]
    y_train = t_input[target].values

    model = make_pipeline(tfidf_transformer, lr)

    #fit model and export
    model.fit(x_train, y_train)
    export_model = f'LOGREG_RELEVANCE/MODELS/{filename}_model.pkl'
    pickle.dump(model, open(path+export_model, 'wb'))

    #print report for user
    y_pred = cross_val_predict(model, x_train, y_train, cv=cv_int)
    report = classification_report(y_train, y_pred)
    print('report:', report, sep='\n')

def lr_predict(path, filename, p_input, p_feature):
    """ Classify text using pickled model created with above function.

    Parameters
    ----------
    p_input:
        input to predict, dataframe (var)
    p_feature:
        df column, values should be string formatted (str)
    filename: 
        model file name to use for classification (str)
    path: 
        parent folder (str/var)
    """
    
    #load model 
    export_model = f'{path}LOGREG_RELEVANCE/MODELS/{filename}_model.pkl'
    model = pickle.load(open(export_model, 'rb'))
    
    #predict and score 
    x_predict = p_input[p_feature]
    y_predict = model.predict(x_predict)
    scores = model.decision_function(x_predict)
    probability = model.predict_proba(x_predict)

    #copy prediction input and add scores 
    result = p_input.copy()
    result['Prediction'] = y_predict
    result['Score'] = scores
    result['Probability'] = probability[:,1]
    result['Input Length'] = result[p_feature].str.len()
    return result