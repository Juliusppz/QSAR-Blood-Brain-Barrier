import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, cohen_kappa_score, matthews_corrcoef, \
    recall_score, f1_score
from sklearn.externals import joblib
import csv
from imblearn.over_sampling import RandomOverSampler


if __name__ == '__main__': # protect for joblib when running on windows
    with open('data/bbb_train.csv', newline='') as csvfile:
        molprediction = csv.reader(csvfile, delimiter=',')
        train = list(molprediction)  # read training file

    train = np.array(train)
    moltraindata = train[1:, 0]
    ytraindata = train[1:, 1]
    ytraindata = ytraindata.astype(int)

    moltraindata = [Chem.MolFromSmiles(mol) for mol in moltraindata]  # get rdkit mol object

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
                                axis=1)  # build x training data from fingerprints and some descriptors

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
    x_final_test = np.concatenate((x_final_test, testdescriptors), axis=1)  # get x test data ready

    print("ratio: " + str(sum(ytraindata) / len(ytraindata)))  # take a look at the data balance

    randomseed = 789

    x_train, x_test, y_train, y_test = train_test_split(xtraindata, ytraindata, test_size=0.25, random_state=randomseed,
                                                        stratify=ytraindata)  # keep 25 % for self-validation
    cv = StratifiedKFold(n_splits=10, random_state=randomseed)  # use stratified 10-fold cross validation

    ros = RandomOverSampler(random_state=randomseed)
    x_train, y_train = ros.fit_sample(x_train, y_train)  # oversample positive samples to accomplish data balance

    scaler = StandardScaler().fit(x_train)  # scale data (not very helpful here, but useful for other models)
    x_train = scaler.transform(x_train)
    joblib.dump(scaler, "mydata/scale_morgan1.pkl", compress=3)

    paramgrid = {
        "max_features": [x_train.shape[1] // 20, x_train.shape[1] // 15, x_train.shape[1] // 10, x_train.shape[1] // 5],
        "n_estimators": [100, 150, 250, 400]}
    m_rf = GridSearchCV(RandomForestClassifier(), paramgrid, n_jobs=2, cv=cv, verbose=1,
                        scoring="accuracy")  # test rf for a grid of params
    m_rf.fit(x_train, y_train)
    joblib.dump(m_rf, "mydata/model_morgan1_rf.pkl", compress=3)

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

    x_final_test = scaler.transform(x_final_test)
    pred_rf_test = m_rf.predict(x_final_test)

    print("testratio: " + str(sum(pred_rf_test) / len(pred_rf_test)))  # the ratios in the testset seem to be different

    moltmp = np.array(testdata)  # get results ready
    pred_rf_test = np.concatenate(([moltmp[1:, 0]], pred_rf_test[:, None].T), axis=0)
    pred_rf_test = np.concatenate(([moltmp[0, :], ["Label"]], pred_rf_test[:, :]), axis=1)

    with open("data/bbb_test_answers.csv", 'w', newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(pred_rf_test.T)  # write results
