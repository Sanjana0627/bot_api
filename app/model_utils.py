import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import joblib
from datetime import datetime

def safe_float(x):
    try:
        return float(x)
    except:
        return 0

def preprocess_data(df):
    df['CREATED_ON'] = pd.to_datetime(df['CREATED_ON'], errors='coerce')
    df['LEAD_AGE_DAYS'] = (pd.to_datetime("today") - df['CREATED_ON']).dt.days

    for col in [
        'NO_OF_CUSTOMER_VISITS', 'TOTAL_TIME_SPENT_AT_CUSTOMER',
        'NO_OF_DEALER_VISITS', 'TOTAL_TIME_SPENT_AT_DEALER'
    ]:
        df[col] = df[col].apply(safe_float)

    df['EFFORT_SCORE'] = (
        df['NO_OF_CUSTOMER_VISITS'].fillna(0) * 3 +
        np.log1p(df['TOTAL_TIME_SPENT_AT_CUSTOMER'].fillna(0)) * 0.8 +
        df['NO_OF_DEALER_VISITS'].fillna(0) * 2 +
        np.log1p(df['TOTAL_TIME_SPENT_AT_DEALER'].fillna(0)) * 0.6
    )

    df = df.fillna("Unknown")
    return df

def train_model():
    df = pd.read_csv('app/Lead_Data.csv')
    df['Converted_Label'] = (df['LEAD_STAGE'].str.lower().str.strip() == 'converted').astype(int)
    df = df[~df['LEAD_STAGE'].str.lower().str.strip().isin(['closed', 'fake', 'lost'])].copy()

    selected_cols = [
        'CREATEDBY', 'CITY', 'DISTRICT_NAME', 'STATE', 'LEAD_TYPE',
        'LEAD_SOURCE', 'LEAD_SOURCE_TYPE', 'EXPECTED_QTY',
        'EFFORT_SCORE', 'LEAD_AGE_DAYS', 'Converted_Label'
    ]

    df = preprocess_data(df)
    df = df[selected_cols]
    X = df.drop(columns=['Converted_Label'])
    y = df['Converted_Label']
    cat_features = X.select_dtypes(include='object').columns.tolist()

    train_pool = Pool(X, y, cat_features=cat_features)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, eval_metric='AUC',
                               random_seed=42, verbose=0, class_weights=[1, 1.5])
    model.fit(train_pool, early_stopping_rounds=50)

    model.save_model('app/models/lead_catboost_model.cbm')
    joblib.dump(cat_features, 'app/models/catboost_cat_features.pkl')
    return "Model trained and saved."

def predict_lead(input_data: dict):
    model = CatBoostClassifier()
    model.load_model('app/models/lead_catboost_model.cbm')
    cat_features = joblib.load('app/models/catboost_cat_features.pkl')

    df = pd.DataFrame([input_data])
    df = preprocess_data(df)

    selected_cols = [
        'CREATEDBY', 'CITY', 'DISTRICT_NAME', 'STATE', 'LEAD_TYPE',
        'LEAD_SOURCE', 'LEAD_SOURCE_TYPE', 'EXPECTED_QTY',
        'EFFORT_SCORE', 'LEAD_AGE_DAYS'
    ]
    df = df[selected_cols]
    df_pool = Pool(df, cat_features=cat_features)
    prob = model.predict_proba(df_pool)[0][1]
    return round(prob, 4)
