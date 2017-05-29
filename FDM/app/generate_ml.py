
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import os
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
from app.learn_ml import *


def draw_sample(experiment, nrcomp, gen_comp_dict):
    population = np.array([comp for comp in gen_comp_dict.keys()])
    probability = np.array([nrmat[0] for nrmat in gen_comp_dict.values()])
    totalcomp = np.sum(probability)
    probability = probability / totalcomp
    sample = np.random.choice(population, size=nrcomp, replace=False, p=probability)
    bom = []
    totalparts = 0.0
    for component in sample:
        mu = gen_comp_dict[component][1]
        std = gen_comp_dict[component][2]
        dosage = stats.truncnorm.rvs(-mu/std, (1-mu)/std, mu, std, size=1)
        totalparts = totalparts + dosage[0]
        if component in models.fdm.comp_names:
            comp_name = models.fdm.comp_names[component]
        else:
            comp_name = ''
        bom.append([component, comp_name, dosage[0]])
# normalize bomparts
    for ix, comp in enumerate(bom):
        bom[ix] = [comp[0], comp[1], comp[2]/totalparts]
    return bom

def generate_ml(nrexp, region_choices, category_choices, corp_choices):

# Read E2E data
    e2e = models.fdm.e2e.copy(deep=True)
    if len(region_choices):
        e2e = e2e[e2e['region'].isin(region_choices)]
    if len(category_choices):
        e2e = e2e[e2e['catclass'].isin(category_choices)]
    if len(corp_choices):
        e2e = e2e[e2e['custcorp'].isin(corp_choices)]

# Determine nr of compoents and dosage distribution in filtered E2E
    matcmp_df = e2e.groupby(['material','component','bompartp']).size().reset_index().rename(columns={0:'count'})
    gen_comp_dict = dict(matcmp_df['component'].value_counts())
    for component in gen_comp_dict.keys():
        dosage = matcmp_df[matcmp_df['component']==component]['bompartp']
        nrcomp = gen_comp_dict[component]
        dosage_mu, dosage_std = stats.norm.fit(dosage)
        gen_comp_dict[component] = [nrcomp, dosage_mu, dosage_std]
# X is the number of components used by materials, based on normal distribution calc components for generated experiments
    nrcomp_X = np.array(matcmp_df['material'].value_counts())
    nrcomp_mu, nrcomp_std = stats.norm.fit(nrcomp_X)
    nrcomp_Y = stats.truncnorm.rvs(1, (1000-nrcomp_mu)/nrcomp_std,
                                   nrcomp_mu, nrcomp_std, size=nrexp+1).astype(int)

# Generate new E2E dataset for new experimentes
    gen_e2e = DataFrame(columns=['material','component','bompartp'])
    models.fdm.generate_li = None # cleanup old data
    models.fdm.generate_li = []
    segment = 'regions:'
    for region in region_choices:
        segment = segment + ' ' + region        
    for expnr in range (1, nrexp+1):
        experiment = 'FDM%0.9d' % expnr
        nrcomp = nrcomp_Y[expnr]
        bom = draw_sample(experiment, nrcomp, gen_comp_dict)
        e2eexp = pd.DataFrame(bom, columns=['component', 'comp_name', 'bompartp'])
        e2eexp['material'] = experiment
        gen_e2e = gen_e2e.append(e2eexp, ignore_index=True)
        gen1 = models.generate_cl(experiment, segment, nrcomp)
        gen1.bom = bom
        models.fdm.generate_li.append(gen1)

# For generated E2E dataset execute the prediction
    nrcomp = len(models.fdm.comp_dict)
    materials = np.unique(gen_e2e.material).tolist()
    nrmat = len(materials)
    mat_dict =  dict([(mat, {'ix' : materials.index(mat)}) for mat in materials])
    Xnw = np.zeros(shape=(nrmat, nrcomp), dtype=float)
#    matcmp_df = DataFrame(columns=models.fdm.components)
#    matcmp_df.index.name = 'material'
#    matcmp_df.columns.name = 'component'
    for idx, row_srs in gen_e2e.iterrows() :
        if row_srs.component in models.fdm.comp_dict.keys():
            matix = mat_dict[row_srs.material]['ix']
            cmpix = models.fdm.comp_dict[row_srs.component]['ix']
            Xnw[matix][cmpix] = row_srs.bompartp
#            matcmp_df.set_value(row_srs.material, row_srs.component, row_srs.bompartp) 
#    matcmp_df.fillna(0, inplace=True)
#    Xnw =  np.array(matcmp_df)
    print('lengths Xnw: ', len(Xnw))
    print('Components/Features Xnw[0].size: ', Xnw[0].size)

    for learn in models.fdm.learn_li:
        Ynw = learn.clf.predict(Xnw)
        print('Region {}, Classifier {}: WINs Yts {}, Ynw {}, accuracy {}: '.
            format(learn.segment, learn.clf_name, learn.y_ts_wins, len(Ynw[[Ynw == 'WIN']]), learn.accuracy))        
        print('Region {0}, Classifier {1}: Accuracy is: {2:5.2f}'.format(learn.segment, learn.clf_name, learn.accuracy))   

        for experiment in mat_dict.keys():
            gen1 = [ exp for exp in models.fdm.generate_li if exp.experiment == experiment][0]
            ix = mat_dict[experiment]['ix']
            prediction1 = models.prediction_cl(material=experiment, descr='', segment = learn.segment, clf_name = learn.clf_name, project='FDM',
                                               fnresstssbm=Ynw[ix], accuracy=learn.accuracy )
            gen1.predictions.append(prediction1)
    return

def predict_gen_ml():
    prediction_li = []
    for gen1 in models.fdm.generate_li:
        prediction_li.extend(gen1.predictions)
    return(prediction_li)

  