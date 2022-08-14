#import relevant modules and libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders.target_encoder import TargetEncoder
import scipy

#Disable Warnings
import warnings
warnings.filterwarnings("ignore")

#Dataset I/O
def read_file(path):
    return pd.read_csv(path)

#Define training loop
def training_loop(estimator,skf,X_train,y_train):
    fold=0
    scores,predictions=[],[]
    for train_index, test_index in skf.split(X_train,y_train):
        fold+=1
        print(f"================Fold:{fold}====================")
        xtrain, xtest = X_train.iloc[train_index],X_train.iloc[test_index]
        ytrain, ytest = y_train.iloc[train_index],y_train.iloc[test_index]
        model = estimator.fit(xtrain,ytrain)
        #predict on train
        pred_train = model.predict(xtrain)
        #Get MAE on training
        score_train = mean_absolute_error(ytrain,pred_train)
        #Predict on test
        pred_test = model.predict(xtest)
        pred_ = model.predict(test_)
        #Get MAE on test
        score_test = mean_absolute_error(ytest,pred_test)
        #Print scores
        print(f"The (train) mean_absolute_error for Fold({fold}): {score_train}")
        print(f"The (test) mean_absolute_error for Fold({fold}): {score_test}\n\n")
        #Store scores and predictions
        scores.append(score_test)
        predictions.append(pred_)
    print(f"Mean MAE score on test set: {np.mean(scores)}")
    return predictions
    
#read training and test datasets
train = read_file(path='train.csv')
test = read_file(path='test.csv')

#Combine both train and test sets for easy wrangling
all_data = pd.concat([train,test]).reset_index(drop=True)

#Feature Engineering
all_data["bank_account"].replace({"Yes":1,"No":0},inplace=True)
all_data["cellphone_access"].replace({"Yes":1,"No":0},inplace=True)
all_data["location_type"].replace({"Rural":0,"Urban":1},inplace=True)


n_train = train.shape[0]
n_test = test.shape[0]
#seperate combine dataset back into test and train categories
train = all_data.iloc[:n_train]
test = all_data.iloc[-n_test:].reset_index(drop=True)

train['bank_account'] = train['bank_account'].astype(np.int64)
target = train['bank_account']
test__ = test.drop(['uniqueid','bank_account'],1)
train.drop(["uniqueid",'bank_account'],1,inplace=True)

#Target encode the categorical features
cat_cols=["country","gender_of_respondent","education_level","job_type","marital_status","relationship_with_head"]
TE = TargetEncoder(verbose=0, cols=cat_cols, drop_invariant=False, return_df=True)
#fit target encoder
TE.fit(train,target)
#transform categorical features using the target encoder
train_ = TE.transform(train)
test_ = TE.transform(test__)

#Standardize skewed numeric variables
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_[['household_size',"age_of_respondent"]] = ss.fit_transform(train_[['household_size',"age_of_respondent"]])
test_[['household_size',"age_of_respondent"]] = ss.transform(test_[['household_size',"age_of_respondent"]])

#Drop redundant features.
columns_to_drop = ['year']
train_.drop(columns_to_drop,1,inplace=True)
test_.drop(columns_to_drop,1,inplace=True)

#Create a hold-out test set (20%), use the other 80% for cross_validation.
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(train_,target,test_size=0.2,stratify=target)

#Due to data imbalance a stratified cross_validation would be a good measure of model perfromance
skf = StratifiedKFold(n_splits=7, random_state=10, shuffle=True)

#Train a regularized greedy tree model
from rgf.sklearn import RGFClassifier
rgf_model = RGFClassifier(n_jobs=-1,algorithm='RGF_Opt')
pred_rgf =training_loop(rgf_model,skf,X_train,y_train)

#Onbtain then mode prediction across all seven cross_validation splits as final prediction.
preds = scipy.stats.mode(np.stack(pred_rgf), axis=0)[0].flatten()


# Create prediction DataFrame
submission = pd.DataFrame({"uniqueid": test["uniqueid"] + " x " + test["country"],
                           "bank_account": preds})

# Create prediction csv file
submission.to_csv('rgf_opt_predictions.csv', index = False)