
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
from FDM.settings import BASE_DIR


def read_components():
    comp_file = os.path.join(BASE_DIR, 'data/components.csv')
    if os.path.exists(comp_file):
        models.fdm.comp_names = {}
        file = open(comp_file, 'r')
        pyfile = File(file)
        for line in pyfile:
            words = line.rstrip().split(',', 1)
            models.fdm.comp_names[str(words[0])] = str(words[1])
        pyfile.close()


def retrieve_components (cat='', corp='', region=''):
#    url="http://sapdgw.iff.com:8000/sap/opu/odata/sap/ZE2EM001_EQ001/AZE2EM001_EQ001(ZVOCCL='"+cat
    url="http://sapqgw.iff.com:8000/sap/opu/odata/sap/ZE2EM001_EQ001/AZE2EM001_EQ001(ZVOCCL='"+cat
#   url="http://sappgw.iff.com:8000/sap/opu/odata/sap/ZE2EM001_EQ001/AZE2EM001_EQ001(ZVOCCL='"+cat
    url=url+"',ZVOCORP='"+corp
    url=url+"',ZVORGN='"+region+"')/Results/?$format=json"

    comp_file = os.path.join(BASE_DIR, 'data/components.csv')
    if os.path.exists(comp_file):
        read_components()
    else:
        resp = requests.get(url, auth=('BWRA', 'sapsapif'))   
        if resp.status_code != 200:
            return([])
          
        data = resp.json()
        q = data['d']['results']
        components_df = DataFrame(q)
        components_df.rename(columns = {'A0COMPONENT':'component', 'A0COMPONENT_T':'comp_name', 'ZCPE2E002_EQ002_BOMPARTP':'bompartp'}, inplace=True)
#       component_names = {comp_key: comp_name for comp_key, comp_name in components_df['component', 'comp_name']}
        models.fdm.comp_names = dict(zip(components_df.component, components_df.comp_name))
        file = open(comp_file, 'w')
        pyfile = File(file)
        for comp_key, comp_name in component_names.items():
            pyfile.write(comp_key + ',' + comp_name + '\n')
        pyfile.close()


def read_exp_e2e():
    print (os.getcwd())
    exp_file = os.path.join(BASE_DIR, "data/ZCRE2E001.txt")
    e2e = pd.read_csv(exp_file, sep=';', encoding='ISO-8859-1', low_memory=False)

    e2e.columns = ('co-area','fiscper', 'material','country','plant',
                   'profit-ctr','salesorg','sold-to','experiment','fiscvarnt',
                   'component','project',
                   'catclass', 'enduse','ccprj', 'ccrsc',   
                   'opptyp', 'oppsts', 'prjown', 'prjtyp', 'prjsts', 'fnressts',
                   'fnstssbm','region', 'country_group',  
                   'custcat', 'custcorp',  
                   'bompartp')
                 
    e2e = e2e.drop(['co-area', 'profit-ctr','fiscvarnt', 'country', 'prjown', 'plant', 'salesorg', 'ccrsc'], axis=1) 

    e2e = e2e[e2e.bompartp > 0.00001]
    e2e.reset_index(drop=True, inplace=True)
    for f in ['custcorp', 'region','component', 'country_group']:
        if isinstance(e2e[f][0], float):
            e2e[f] = e2e[f].map('{:.0f}'.format)
    for f in ['catclass']:
        if isinstance(e2e[f][0], float):
            e2e[f] = e2e[f].map('{:03.0f}'.format)
    for f in ['fiscper']:
        if isinstance(e2e[f][0], np.integer):
            e2e[f] = e2e[f].map('{:0d}'.format)                                
    e2e['component'] =  e2e['component'].str[-8:]
    e2e.loc[pd.isnull(e2e.material), 'material'] = e2e.experiment
    e2e.loc[e2e.fnstssbm != 'WIN', 'fnstssbm'] = 'LOSS'
    e2e[['fnressts','fnstssbm']][e2e.fnressts == 'WIN']
    return e2e

def read_subm_e2e():
    print (os.getcwd())
    subm_file = os.path.join(BASE_DIR, "data/ZCPE2E002.txt")
    e2e = pd.read_csv(subm_file, sep=';', encoding='ISO-8859-1', low_memory=False)

    e2e.columns = ('fiscper', 'catclass', 'enduse', 'prodhier', 'bomlvl', 'expcmp',
                      'exptyp', 'co-area', 'profit-ctr', 'fiscvarnt', 'ccprj', 'ccrsc', 
                     'opptyp', 'oppsts', 'prjown', 'prjtyp', 'prjsts', 'fnressts',
                     'project', 'region', 'country_group', 'country', 
                     'custcat', 'custcorp', 'sold-to', 'npidoc', 'npityp', 'subm',
                     'material', 'fnstssbm', 'experiment', 'perfumer', 'component', 
                     'adoptions', 'projects', 'submissions', 'annpotval',
                     'annpotvol', 'bomparts', 'bompartp', 'items', 'base_uom', 
                     'doc_currcy', 'annpot_currcy', 'annpot_uom')
                 
 
    e2e = e2e.drop(['prodhier', 'bomlvl', 'expcmp', 'exptyp', 'co-area', 'profit-ctr',
                        'fiscvarnt', 'prjown','npidoc', 'npityp'], axis=1)  
    e2e = e2e.drop(['adoptions', 'projects', 'submissions', 'annpotval', 'annpotvol', 'bomparts',
                        'items', 'base_uom', 'doc_currcy', 'annpot_currcy', 'annpot_uom'], axis=1)    

    e2e = e2e[e2e.bompartp > 0.00001]
    e2e.reset_index(drop=True, inplace=True)
    for f in ['custcorp', 'region','component', 'country_group', 'country', 'enduse']:
        if isinstance(e2e[f][0], float):
            e2e[f] = e2e[f].map('{:.0f}'.format)
    for f in ['catclass']:
        if isinstance(e2e[f][0], float):
            e2e[f] = e2e[f].map('{:03.0f}'.format)
    for f in ['fiscper', 'perfumer']:
        if isinstance(e2e[f][0], np.integer):
            e2e[f] = e2e[f].map('{:0d}'.format)                                 
    e2e['component'] =  e2e['component'].str[-8:]
    e2e['sold-to'] =  e2e['sold-to'].str.lstrip("0")
    e2e['material'] =  e2e['material'].str.lstrip("0")
    e2e.loc[pd.isnull(e2e.material), 'material'] = e2e.experiment
    e2e.loc[e2e.fnstssbm != 'WIN', 'fnstssbm'] = 'LOSS'
    e2e[['fnressts','fnstssbm']][e2e.fnressts == 'WIN']
    return e2e

def retrieve_ml(e2e_prediction):

    retrieve_components()
    if e2e_prediction == 'e2eexp':
        e2e = read_exp_e2e()
    else:
        e2e = read_subm_e2e()
    models.fdm.e2e = e2e
  