
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import requests
import os
from django.core.files import File
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import app.models as models

    #names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
    #         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
    #         "Quadratic Discriminant Analysis"]
    #
    #classifiers = [
    #    KNeighborsClassifier(3),
    #    SVC(kernel="linear", C=0.025),
    #    SVC(gamma=2, C=1),
    #    DecisionTreeClassifier(max_depth=5),
    #    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #    AdaBoostClassifier(),
    #    GaussianNB(),
    #    LinearDiscriminantAnalysis(),
    #    QuadraticDiscriminantAnalysis()]


def calc_sample_weight(segment_df, win_weight):
    if win_weight == 0.0:
        prjmat_df = segment_df.groupby(['project','material','fnstssbm']).size().reset_index().rename(columns={0:'count'})
        nrwin = sum(prjmat_df['fnstssbm'] == 'WIN')
        nrloss = sum(prjmat_df['fnstssbm'] == 'LOSS')
        if nrwin > 0:
            ww = nrloss / nrwin
        else:
            ww = 1.0
        lw = 1.0
    elif win_weight == 1.0:
        ww = 1.0
        lw = 1.0
    else:
        ww = win_weight
        lw = 1.0
    return ww, lw


def learn_ml(ml_choices, ms_choices, test_perc, win_weight):

    e2e = models.fdm.e2e
                 
    models.fdm.components = np.unique(e2e.component).tolist()
    comp_counts_s = e2e['component'].value_counts()
    models.fdm.comp_dict = dict([(comp, {'ix' : models.fdm.components.index(comp)}) for comp in models.fdm.components])
    for comp in comp_counts_s.index:
        models.fdm.comp_dict[comp]['count'] = comp_counts_s[comp]
    nrcomp = len(models.fdm.comp_dict)

    models.fdm.learn_li = []
    for ms_choice in ms_choices:
        for segment, segment_df in e2e.groupby (ms_choice) :
            print ('Segment key ', ms_choice, segment)
            if segment != 'nan':
                materials = np.unique(segment_df.material).tolist()
                nrmat = len(materials)
                mat_dict =  dict([(mat, {'ix' : materials.index(mat)}) for mat in materials])
                ww, lw = calc_sample_weight(segment_df, win_weight)
#                columns = models.fdm.components
#                columns.extend(['fnstssbm','sample_weight'])
#                matcmp_df = DataFrame(columns=columns)
#                matcmp_df.index.name = 'material'
#                matcmp_df.columns.name = 'component'
#                matcmp_df = segment_df.pivot_table(['bompartp'], index=['material'], columns=['component'], fill_value=0)
                X = np.zeros(shape=(nrmat, nrcomp), dtype=float)
                sample_weight = np.zeros(shape=(nrmat), dtype=float)
                Y = np.zeros(shape=(nrmat), dtype=object)
                for idx, row_srs in segment_df.iterrows() :
#                    matcmp_df.set_value(row_srs.material, 'fnstssbm', row_srs.fnstssbm)
#                    matcmp_df.set_value(row_srs.material, row_srs.component, row_srs.bompartp)
#                    if row_srs.fnstssbm == 'WIN':
#                        matcmp_df.set_value(row_srs.material, 'sample_weight', ww)
#                    else:
#                        matcmp_df.set_value(row_srs.material, 'sample_weight', lw)
                    matix = mat_dict[row_srs.material]['ix']
                    cmpix = models.fdm.comp_dict[row_srs.component]['ix']
                    X[matix][cmpix] = row_srs.bompartp
                    sample_weight[matix] = ww if row_srs.fnstssbm == 'WIN' else lw
                    Y[matix] = row_srs.fnstssbm
#                matcmp_df.fillna(0, inplace=True)
#                Y =  np.array(matcmp_df.fnstssbm)
#                sample_weight = np.array(matcmp_df.sample_weight)
#                X =  np.array(matcmp_df.drop(['fnstssbm','sample_weight'], axis=1))
#               msk = np.random.rand(len(Y)) < (1-test_perc)/100.0
#               Xtr = X[msk]
#               Xtr = X[~msk]
                Xtr, Xts, Ytr, Yts, sample_weigth_tr, sample_weight_ts = train_test_split(X, Y, sample_weight, test_size=test_perc/100.0, random_state=0)
                print('lengths Xtr, Xts, Ytr, Yts, sample_weight_tr, sample_weight_ts: ', len(Xtr), len(Xts), len(Ytr), len(Yts), len(sample_weigth_tr), len(sample_weight_ts))
                print('Components/Features X[0].size: ', X[0].size)
                print('Components/Features Xtr[0].size: ', Xtr[0].size)


                for ml_choice in ml_choices:
                    name = models.ML[ml_choice][0]
                    clf = models.ML[ml_choice][1]
                    if ml_choice in ['svm', 'logit', 'nn', 'bayes']:
                        clf.fit(X, Y, sample_weight=sample_weight)
                    else:
                        clf.fit(X, Y)
                    Ypr = clf.predict(Xts)
                    print('Segment {}, Classifier {}: WINs Yts {}, Ypr {}: '.
                        format(segment, name, len(Yts[[Yts == 'WIN']]), len(Ypr[[Ypr == 'WIN']])))        
                    print('Segment {}, Classifier {}: Misclassified samples: {}'.format(segment, name, (Yts != Ypr).sum()))
                    print('Segment {0}, Classifier {1}: Accuracy is: {2:5.2f}'.format(segment, name, accuracy_score(Yts, Ypr) ))   
                    learn1 = models.learn_cl(segment=segment, clf_name=name, clf=clf,
                                      y_ts_wins=len(Yts[[Yts == 'WIN']]), y_pr_wins=len(Ypr[[Ypr == 'WIN']]),
                                      y_ts_loss=len(Yts[[Yts == 'LOSS']]), y_pr_loss=len(Ypr[[Ypr == 'LOSS']]),
                                      accuracy=accuracy_score(Yts, Ypr))
                    models.fdm.learn_li.append(learn1)

    return
  