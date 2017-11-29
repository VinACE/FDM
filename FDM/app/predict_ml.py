

# Prediction is called with the experiment, or submission to predict. This prediciton works in the next steps
# 1. the formula of the experiment / submission is retrieved
# 2. next the formula is predicted for the different markete segments
# 3. the results are captured in a predictresults

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import requests
import app.models as models
from .learn_ml import *

def predict_exp_ml (project, experiment, submission):

#    url="http://sapdgw.iff.com:8000/sap/opu/odata/sap/ZCRE2E001_EQ002/AZCRE2E001_EQ002(ZVOPROJ='"+project
    url="http://sapqgw.iff.com:8000/sap/opu/odata/sap/ZCRE2E001_EQ002/AZCRE2E001_EQ002(ZVOPROJ='"+project
    url=url+"',ZVOEXPNUM='"+experiment
    url=url+"',VOMAT='"+submission+"')/Results/?$format=json"

    resp = requests.get(url, auth=('BWRA', 'sapsapif'))
    if resp.status_code != 200:
        return([])    
               
    data = resp.json()
    q = data['d']['results']
    
    df = DataFrame(q)
    expnums = df['ZEXPNUM'].unique()
    prediction_li = []
    for i in range(0, len(expnums)):
        project = df[df['ZEXPNUM'] == expnums[i]]['ZPROJECT'].iloc[0]
        prediction1 = prediction_cl(experiment=expnums[i], descr='', segment = '10', clf_name = '', project=project, fnresstssbm='LOSS')
        prediction2 = prediction_cl(experiment=expnums[i], descr='', segment = '20', clf_name = '', project=project, fnresstssbm='WIN')
        prediction3 = prediction_cl(experiment=expnums[i], descr='', segment = '30', clf_name = '', project=project, fnresstssbm='LOSS')
        prediction4 = prediction_cl(experiment=expnums[i], descr='', segment = '40', clf_name = '', project=project, fnresstssbm='LOSS')
        prediction_li.append(prediction1)
        prediction_li.append(prediction2)
        prediction_li.append(prediction3)
        prediction_li.append(prediction4)

    #prediction_li = []
    #matcmp_df = DataFrame(columns=components)
    #matcmp_df.index.name = 'material'
    #matcmp_df.columns.name = 'component'
    #for idx, row_srs in df.iterrows() :
    #    matcmp_df.set_value(row_srs.material, row_srs.component, row_srs.bompartp) 
    #matcmp_df.fillna(0, inplace=True)
    #X =  np.array(matcmp_df, axis=1)
    #print('lengths X: ', len(X))

    #for name, clf in zip(names, classifiers):
    #    Y = clf.predict(X)
    #    print('Classifier {}: WINs Y {}: '.
    #        format(name, len(Y[[Y == 'WIN']]))        

    #    prediction1 = prediction_cl(segment=region, clf_name=name, clf=clf,
    #                        y_ts_wins=len(Yts[[Yts == 'WIN']]), y_pr_wins=len(Ypr[[Ypr == 'WIN']]),
    #                        accuracy=accuracy_score(Yts, Ypr))
    #    prediction_li.append(learn1)


    return(prediction_li)


def retrieve_exp_ofl (project, experiment, submission):
    if models.fdm.e2e.empty == True:
        models.fdm.e2e = read_exp_e2e()

    e2e = models.fdm.e2e
    if len(project):
        pre_e2e = e2e[e2e['project'] == project]
    if len(submission):
        pre_e2e = e2e[e2e['submission'] == submission]
    if len(experiment):
        pre_e2e = e2e[e2e['experiment'] == experiment]
    return pre_e2e


def retrieve_exp_onl (project, experiment, submission):
#   url="http://sapdgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCRE2E001_EQ002(ZVOPROJ='"+project
#   url="http://sapqgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCRE2E001_EQ002(ZVOPROJ='"+project
    url="http://sappgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCRE2E001_EQ002(ZVOPROJ='"+project
    url=url+"',ZVOEXPNUM='"+experiment
    url=url+"',VOMAT='"+submission+"')/Results/?$format=json"

    resp = requests.get(url, auth=('BWRA', 'sapsapif'))   
    if resp.status_code != 200:
        return([])
          
    data = resp.json()
    q = data['d']['results']
    pre_e2e = DataFrame(q)
    pre_e2e.rename(columns = {'A0MATERIAL':'material', 'A0COMPONENT':'component', 'ZCPE2E002_EQ002_BOMPARTP':'bompartp'}, inplace=True)
    return pre_e2e


def retrieve_subm_ofl (project, experiment, submission):
    if models.fdm.e2e.empty == True:
        models.fdm.e2e = read_subm_e2e()

    e2e = models.fdm.e2e
    if len(project):
        pre_e2e = e2e[e2e['project'] == project]
    if len(submission):
        pre_e2e = e2e[e2e['submission'] == submission]
    if len(experiment):
        pre_e2e = e2e[e2e['experiment'] == experiment]
    return pre_e2e


def retrieve_subm_onl (project, experiment, submission):
#   url="http://sapdgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCPE2E002_EQ002(ZVOPROJ='"+project
#   url="http://sapqgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCPE2E002_EQ002(ZVOPROJ='"+project
    url="http://sappgw.iff.com:8000/sap/opu/odata/sap/ZCPE2E002_EQ002/AZCPE2E002_EQ002(ZVOPROJ='"+project
    url=url+"',ZVOEXPNUM='"+experiment
    url=url+"',VOMAT='"+submission+"')/Results/?$format=json"

    resp = requests.get(url, auth=('BWRA', 'sapsapif'))   
    if resp.status_code != 200:
        return([])
          
    data = resp.json()
    q = data['d']['results']
    pre_e2e = DataFrame(q)
    pre_e2e.rename(columns = {'A0MATERIAL':'material', 'A0COMPONENT':'component',
                              'ZCPE2E002_EQ002_BOMPARTP':'bompartp', 'ZCPE2E002_EQ002_BOMPARTS':'bomparts'}, inplace=True)
    pre_e2e['bompartp'] = [float(f) for f in pre_e2e['bompartp']]
    pre_e2e['bomparts'] = [float(f) for f in pre_e2e['bomparts']]
    return pre_e2e


def predict_e2e (pre_e2e, project):
    prediction_li = []

    nrcomp = len(models.fdm.comp_dict)
    materials = np.unique(pre_e2e.material).tolist()
    nrmat = len(materials)
    mat_dict =  dict([(mat, {'ix' : materials.index(mat)}) for mat in materials])
    Xnw = np.zeros(shape=(nrmat, nrcomp), dtype=float)

#    matcmp_df = DataFrame(columns=models.fdm.components)
#    components_set = set(models.fdm.components)
#    matcmp_df.index.name = 'material'
#    matcmp_df.columns.name = 'component'
    for idx, row_srs in pre_e2e.iterrows() :
        if row_srs.component in models.fdm.comp_dict.keys():
            matix = mat_dict[row_srs.material]['ix']
            cmpix = models.fdm.comp_dict[row_srs.component]['ix']
            Xnw[matix][cmpix] =  row_srs.bompartp
#            matcmp_df.set_value(row_srs.A0MATERIAL, row_srs.A0COMPONENT, row_srs.ZCPE2E002_EQ002_BOMPARTP) 
#    Xnw =  np.array(matcmp_df)
    print('lengths Xnw: ', len(Xnw))
    print('Components/Features Xnw[0].size: ', Xnw[0].size)

    for learn in models.fdm.learn_li:
        Ynw = learn.clf.predict(Xnw)
        print('Region {}, Classifier {}: WINs Yts {}, Ynw {}, accuracy {}: '.
            format(learn.segment, learn.clf_name, learn.y_ts_wins, len(Ynw[[Ynw == 'WIN']]), learn.accuracy))        
        print('Region {0}, Classifier {1}: Accuracy is: {2:5.2f}'.format(learn.segment, learn.clf_name, learn.accuracy))   

        for material in mat_dict.keys():
            if 'A0MATERIAL_T' in pre_e2e.columns:
                descr = pre_e2e[pre_e2e['material'] == material]['A0MATERIAL_T'].iloc[0]
            else:
                descr = material
            prediction1 = models.prediction_cl(material=material, descr=descr, segment = learn.segment, clf_name = learn.clf_name, project=project,
                                               fnresstssbm=Ynw[mat_dict[material]['ix']], accuracy=learn.accuracy )
            prediction_li.append(prediction1)

    return(prediction_li)


def predict_ml (e2e_prediction, project, experiment, submission):
    if e2e_prediction == 'e2eexp_onl':
        pre_e2e = retrieve_exp_onl(project, experiment, submission)
    elif e2e_prediction == 'e2esubm_onl':
        pre_e2e = retrieve_subm_onl(project, experiment, submission)
    elif e2e_prediction == 'e2eexp_ofl':
        pre_e2e = retrieve_exp_ofl(project, experiment, submission)
    elif e2e_prediction == 'e2esubm_ofl':
        pre_e2e = retrieve_subm_ofl(project, experiment, submission)

    prediction_li = predict_e2e(pre_e2e, project)
    return prediction_li


def predict_df (prediction_li):
    ml_choices = set()
    for p in prediction_li:
        ml_choices.add(p.clf_name)
    ml_choices = list(ml_choices)

    df = pd.DataFrame(index=[[],[]], columns=['material','descr','segment'] + ml_choices)
    for p in prediction_li:
        d = {'material': p.material, 'segment': p.segment, p.clf_name: p.fnresstssbm+' (%.2f)' % p.accuracy}
        s = pd.Series(d)
        df.set_value((p.material, p.segment),'material', p.material)
        df.set_value((p.material, p.segment),'descr', p.descr)
        df.set_value((p.material, p.segment),'segment', p.segment)
        df.set_value((p.material, p.segment),p.clf_name, p.fnresstssbm+' (%.2f)' % p.accuracy)
    df.sort_values(['material', 'segment'], inplace=True)
    df.fillna('', inplace=True)
    return df      

        

        

