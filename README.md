# QSAR: Predicting Blood-Brain Barrier Passage
This is an example of a quantitative structure-activity relationship (QSAR) model. It is build to predict the probability of a substance being able to pass the blood-brain barrier (BBB) using Morgan fingerprints and a number of hand-picked descriptors. The input consists of molecular structures in SMILES format and a random forest classifier is used as a binary classification model. The best hyperparameters are determined by a grid search and the model is validated using 10-fold cross-validation. In the end it is tested on a separate set and used to infer the behavior of unlabeled data.

## Implementation
Numpy is used for handling the data arrays, rdkit for generating the fingerprints as input for the models and sklearn for implementing the models. csv is used to read the SMILES strings from the data files and imblearn offers a simple way to restore the class balance (in this case using random oversampling).
```
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, \
cohen_kappa_score, matthews_corrcoef, recall_score, f1_score
from sklearn.externals import joblib
import csv
from imblearn.over_sampling import RandomOverSampler
```

If the code is to be run on Windows, the main loop has to be protected for parallel execution by joblib. This can be done by running everything after the imports within `if __name__ == '__main__':`.

## Data Preparation
First, the raw training data is loaded and stored in numpy arrays.
```
with open('data/bbb_train.csv', newline='') as csvfile:
    molprediction = csv.reader(csvfile, delimiter=',')
    train = list(molprediction)  

train = np.array(train)
moltraindata = train[1:, 0]
ytraindata = train[1:, 1]
ytraindata = ytraindata.astype(int)
```

Next, the SMILES strings that represent the molecules in the "x" component of the training data are converted into rdkit molecule objects. Using those, certain descriptors and Morgen fingerprints are calculated and are later used to train this model.
```
moltraindata = [Chem.MolFromSmiles(mol) for mol in moltraindata]  

descriptors = [[Descriptors.TPSA(mol), Descriptors.MolLogP(mol), Descriptors.NHOHCount(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), Descriptors.NumHeteroatoms(mol),
                Descriptors.Kappa1(mol), Descriptors.Kappa2(mol), Descriptors.Kappa3(mol),
                Descriptors.NOCount(mol), Descriptors.Chi3n(mol), Descriptors.Chi3v(mol),
                Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol)] for mol in moltraindata]

descriptors = np.asarray(descriptors)

fptraindata = [AllChem.GetMorganFingerprintAsBitVect(mol, 3) for mol in moltraindata]
xtraindata = np.array(fptraindata)
xtraindata = np.concatenate((xtraindata, descriptors),
                            axis=1)  
```

Now the final test data is loaded in the same way as the training data and after training the model, their classes will be inferred. These could be the predictions for new molecules one is interested in examining in the future.
```
with open('data/bbb_test.csv', newline='') as csvfile:
    moltest = csv.reader(csvfile, delimiter=',')
    testdata = list(moltest)  # read test file

moltestdata = np.array(testdata)
moltestdata = moltestdata[1:, 0]

moltestdata = [Chem.MolFromSmiles(mol) for mol in moltestdata]

testdescriptors = [[Descriptors.TPSA(mol), Descriptors.MolLogP(mol), Descriptors.NHOHCount(mol),
                    Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol), Descriptors.NumHeteroatoms(mol),
                    Descriptors.Kappa1(mol), Descriptors.Kappa2(mol), Descriptors.Kappa3(mol),
                    Descriptors.NOCount(mol), Descriptors.Chi3n(mol), Descriptors.Chi3v(mol),
                    Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumAromaticRings(mol),
                    Descriptors.FractionCSP3(mol), Descriptors.NumRotatableBonds(mol)] for mol in moltestdata]

testdescriptors = np.asarray(testdescriptors)

testfptraindata = [AllChem.GetMorganFingerprintAsBitVect(mol, 3) for mol in moltestdata]
x_final_test = np.array(testfptraindata)
x_final_test = np.concatenate((x_final_test, testdescriptors), axis=1)  
```

Here you can take a quick look at the data balance to decide how to proceed further. In this case the data is significantly unbalanced.
```
print("ratio: " + str(sum(ytraindata) / len(ytraindata)))  
```

## Model Setup and Training
The training data is split into the actual training set and a test set. The test set will later be used in addition to validation to test the generality of the model. A 10-fold stratified cross-validation is chosen, the training data is randomly oversampled to restore balance and the training data is scaled to unit variance (after the mean is removed). The scale is stored for later use. Note that the scaling is not required in this case, since we are working with a random forest model. However, for other models this preprocessing step is important.
```
randomseed = 789 

x_train, x_test, y_train, y_test = train_test_split(xtraindata, ytraindata, test_size=0.25, random_state=randomseed,
                                                    stratify=ytraindata)  
cv = StratifiedKFold(n_splits=10, random_state=randomseed) 

ros = RandomOverSampler(random_state=randomseed)
x_train, y_train = ros.fit_sample(x_train, y_train) 

scaler = StandardScaler().fit(x_train)  
x_train = scaler.transform(x_train)
joblib.dump(scaler, "mydata/scale_morgan1.pkl", compress=3)
```

A grid of hyperparameters in terms of the number of decision trees and their maximum number of features is constructed and for each combination a model is generated. After training the models are saved.
```
paramgrid = {
    "max_features": [x_train.shape[1] // 20, x_train.shape[1] // 15, x_train.shape[1] // 10, x_train.shape[1] // 5],
    "n_estimators": [100, 150, 250, 400]}
m_rf = GridSearchCV(RandomForestClassifier(), paramgrid, n_jobs=2, cv=cv, verbose=1,
                    scoring="accuracy") 
m_rf.fit(x_train, y_train)
joblib.dump(m_rf, "mydata/model_morgan1_rf.pkl", compress=3)
```

The scale and the models are loaded and a number of performance metrics of the models are examined using the test data set.
```
scaler = joblib.load("mydata/scale_morgan1.pkl")
m_rf = joblib.load("mydata/model_morgan1_rf.pkl")

print("best parameters: " + str(m_rf.best_params_))
print("grid scores: " + str(m_rf.cv_results_['mean_test_score']))
print(m_rf.cv_results_['params'])

x_test = scaler.transform(x_test)
pred_rf = m_rf.predict(x_test)
print("accuracy: " + str(accuracy_score(y_test, pred_rf)))
print("recall: " + str(recall_score(y_test, pred_rf)))
print("precision: " + str(precision_score(y_test, pred_rf)))
print("f1: " + str(f1_score(y_test, pred_rf)))
print("kappa: " + str(cohen_kappa_score(y_test, pred_rf)))
print("conf. matrix: " + str(confusion_matrix(y_test, pred_rf)))
print("matth. cor.: " + str(matthews_corrcoef(y_test, pred_rf)))
```

The input data for the final predictions is scaled like the training data and their ability to pass the blood brain barrier is inferred from the best model. It can be seen that the ratio for this test set is different than for the training set, which indicates that this data does not come from the same distribution as the training data.
```
x_final_test = scaler.transform(x_final_test)
pred_rf_test = m_rf.predict(x_final_test)

print("testratio: " + str(sum(pred_rf_test) / len(pred_rf_test)))  
```

In the last step, the predictions are brought to the same format as the training data and is written to a file.
```
moltmp = np.array(testdata)
pred_rf_test = np.concatenate(([moltmp[1:, 0]], pred_rf_test[:, None].T), axis=0)
pred_rf_test = np.concatenate(([moltmp[0, :], ["Label"]], pred_rf_test[:, :]), axis=1)

with open("data/bbb_test_answers.csv", 'w', newline='') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(pred_rf_test.T)
```
