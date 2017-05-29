
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

import app.models as models
import app.predict_ml as predict_ml


iX = {
    'bompartp'      : 0,
    'bomparts'      : 1,
    'nr_ingr'       : 2,
    'mnc'           : 3,
    'cut_off'       : 4,
    'X'             : 5,
    'Y'             : 6,
    'odor'          : 7,
    'slope'         : 8,
    'perc_iff'      : 9,
    'perc_pref'     : 10,
    'perc_iff_q'    : 9,
    'perc_pref_q'   : 10,
    'perc_iff_n'    : 11,
    'perc_pref_n'   : 12,
    }


def retrieve(input_file, data_list_file, be):
    global iX

    ml_file = 'data/' + input_file
    models.gns_e2e = pd.read_csv(ml_file, sep=';', encoding='ISO-8859-1', low_memory=False)
    ml_file = 'data/' + data_list_file
    models.g_data_file = pd.read_csv(ml_file, sep=';', encoding='ISO-8859-1', low_memory=False)

    models.gns_e2e[['component', 'material']].fillna('', inplace=True)
    models.gns_e2e.fillna(0, inplace=True)
    for f in ['component', 'material']:
        if isinstance(models.gns_e2e[f][0], float):
            models.gns_e2e[f] = models.gns_e2e[f].map('{:.0f}'.format)
    models.gns_e2e['component'] = ['{0:08d}'.format(int(c)) for c in models.gns_e2e['component']]
    models.g_data_file['bid'].fillna('', inplace=True)
    models.g_data_file[['component', 'comp_neat','comp_repl', 'cid', 'bid']].fillna('', inplace=True)
    models.g_data_file.fillna(0, inplace=True)
    for f in ['component', 'comp_neat','comp_repl', 'cid', 'bid']:
        if isinstance(models.g_data_file[f][0], float):
            models.g_data_file[f] = models.g_data_file[f].map('{:.0f}'.format)
    models.g_data_file['component'] = ['{0:08d}'.format(int(c)) for c in models.g_data_file['component']]
    models.g_data_file.index = models.g_data_file['component']

    ## CLEANUP
    # Delete Components with cut_off equal to zero
    models.g_data_file = models.g_data_file[models.g_data_file.cut_off > 0]
    # Delete input for which no component is defined
    mask = [comp in models.g_data_file.index for comp in models.gns_e2e.component]
    models.gns_e2e = models.gns_e2e[mask]

    #models.g_components = np.unique(models.gns_e2e.component).tolist()
    #comp_counts_s = models.gns_e2e['component'].value_counts()
    models.g_components = np.unique(models.g_data_file.component).tolist()
    models.g_comp_dict = dict([(comp, {'ix' : models.g_components.index(comp)}) for comp in models.g_components])

def prepare(e2e):
    global iX

    nrcomp = len(models.g_comp_dict)
    materials = np.unique(e2e.material).tolist()
    models.g_mat_dict =  dict([(mat, {'ix' : materials.index(mat)}) for mat in materials])
    nrmat = len(materials)

    models.g_X = np.full(shape=(iX['nr_ingr']+1, nrmat, nrcomp), fill_value=np.nan, dtype=np.double)
    models.g_C = np.full(shape=(len(iX), nrcomp), fill_value=np.nan, dtype=np.double)
    models.g_M = np.full(shape=(len(iX), nrmat), fill_value=np.nan, dtype=np.double)
    models.g_X[iX['nr_ingr']] = 0
    models.g_C[iX['nr_ingr']] = 0
    models.g_M[iX['nr_ingr']] = 0
    for idx, row_srs in e2e.iterrows():
        matix = models.g_mat_dict[row_srs.material]['ix']
        if len(models.g_selected) > 0:
            selected = row_srs.component in models.g_selected
        else:
            selected = True
        if not selected:
            continue
        if row_srs.component in models.g_comp_dict:
            cmpix = models.g_comp_dict[row_srs.component]['ix']
            bomparts = models.g_X[iX['bomparts']][matix][cmpix]
            if np.isnan(bomparts):
                bomparts = row_srs.bomparts
            else:
                bomparts = bomparts + row_srs.bomparts
            models.g_X[iX['bomparts']][matix][cmpix] = bomparts
            models.g_X[iX['nr_ingr']][matix][cmpix] = 1
        else:
            models.g_unknown_ipc_l.append(row_srs.component)

    for mat in models.g_mat_dict.keys():
        matix = models.g_mat_dict[mat]['ix']
        bomparts = models.g_X[iX['bomparts'],matix]
        Q =  np.nansum(bomparts)
        if Q > 0:
            factor = 1000.0 / Q
            models.g_X[iX['bompartp'],matix] = models.g_X[iX['bomparts'],matix] * factor

    for comp in models.g_comp_dict.keys():
        cmpix = models.g_comp_dict[comp]['ix']
        nr_ingr = np.sum(models.g_X[iX['nr_ingr'],:,cmpix])
        if len(models.g_selected) > 0:
            selected = comp in models.g_selected
        else:
            selected = True
        models.g_comp_dict[comp]['nr_ingr'] = nr_ingr
        if nr_ingr == 0 or not selected:
            continue
        cid = models.g_data_file.ix[comp].cid
        bid = models.g_data_file.ix[comp].bid
        comp_name = models.g_data_file.ix[comp].comp_name
        models.g_comp_dict[comp]['cid'] = cid
        models.g_comp_dict[comp]['bid'] = bid
        models.g_comp_dict[comp]['comp_name'] = comp_name
        models.g_comp_dict[comp]['cid_name'] = models.g_data_file.ix[comp].cid_name
        if cid in ['777777', '888888', '999999']:
            ingrs = [(comp, comp_name)]
            models.g_comp_dict[comp]['cid'] = models.g_data_file.ix[comp].bid
        else:
            ingrs = [(ingr, models.g_data_file.ix[ingr].comp_name) for ingr in models.g_data_file[models.g_data_file.cid == cid]['component'].tolist()]
        models.g_comp_dict[comp]['ingrs'] = ingrs
        #mask = [[models.g_X[i][cmpix] > 0 for i in range(0,nrmat-1)]]
        #bompartp = np.array([models.g_X[mask, cmpix]])
        bompartp = models.g_X[iX['bompartp'],:,cmpix]
        bomparts = models.g_X[iX['bomparts'],:,cmpix]
        Q = np.nansum(bompartp)
        models.g_comp_dict[comp]['bompartp']    = Q
        models.g_comp_dict[comp]['nr_ingr']     = nr_ingr
        models.g_comp_dict[comp]['min']         = np.nanmin(bompartp)
        models.g_comp_dict[comp]['max']         = np.nanmax(bompartp)
        models.g_comp_dict[comp]['avg']         = np.nanmean(bompartp)
        models.g_comp_dict[comp]['std']         = np.nanstd(bomparts)
        models.g_comp_dict[comp]['mnc']         = Q * models.g_data_file.ix[comp].mnc / 1000.0
        models.g_comp_dict[comp]['cut_off']     = Q / models.g_data_file.ix[comp].cut_off
        models.g_comp_dict[comp]['X']           = Q * models.g_data_file.ix[comp].x_round
        models.g_comp_dict[comp]['Y']           = Q * models.g_data_file.ix[comp].y_round
        models.g_comp_dict[comp]['odor']        = Q * models.g_data_file.ix[comp].odor_strength_index_round
        models.g_comp_dict[comp]['slope']       = Q * models.g_data_file.ix[comp].pp_slope_round

        models.g_C[iX['bompartp'], cmpix] = Q
        models.g_C[iX['nr_ingr'], cmpix] = np.sum(models.g_X[iX['nr_ingr'],:,cmpix])
        models.g_C[iX['mnc'], cmpix] = models.g_data_file.ix[comp].mnc
        models.g_C[iX['cut_off'], cmpix] = models.g_data_file.ix[comp].cut_off
        models.g_C[iX['X'], cmpix] = models.g_data_file.ix[comp].x_round
        models.g_C[iX['Y'], cmpix] = models.g_data_file.ix[comp].y_round
        models.g_C[iX['odor'], cmpix] = models.g_data_file.ix[comp].odor_strength_index_round
        models.g_C[iX['slope'], cmpix] = models.g_data_file.ix[comp].pp_slope_round
        models.g_C[iX['perc_iff'], cmpix] = models.g_data_file.ix[comp].perc_iff
        models.g_C[iX['perc_pref'], cmpix] = models.g_data_file.ix[comp].perc_pref

    for mat, value in models.g_mat_dict.items():
        matix = models.g_mat_dict[mat]['ix']
        bompartp = models.g_X[iX['bompartp'],matix]
        Q = np.nansum(bompartp)
        models.g_mat_dict[mat]['bompartp']  = Q
        models.g_mat_dict[mat]['nr_ingr']   = np.sum(models.g_X[iX['nr_ingr'],matix])
        models.g_mat_dict[mat]['min']       = np.nanmin(bompartp)
        models.g_mat_dict[mat]['max']       = np.nanmax(bompartp)
        models.g_mat_dict[mat]['avg']       = np.nanmean(bompartp)
        models.g_mat_dict[mat]['std']       = np.nanstd(bompartp)
        models.g_mat_dict[mat]['mnc']       = np.nansum(bompartp * models.g_C[iX['mnc']]) / 1000.0
        models.g_mat_dict[mat]['cut_off']   = np.nansum(np.divide(bompartp, models.g_C[iX['cut_off']]))
        models.g_mat_dict[mat]['X']         = np.nansum(bompartp * models.g_C[iX['X']])
        models.g_mat_dict[mat]['Y']         = np.nansum(bompartp * models.g_C[iX['Y']])
        models.g_mat_dict[mat]['odor']      = np.nansum(bompartp * models.g_C[iX['odor']])
        models.g_mat_dict[mat]['slope']     = np.nansum(bompartp * models.g_C[iX['slope']])
        models.g_mat_dict[mat]['perc_iff_n']  = np.count_nonzero(bompartp * models.g_C[iX['perc_iff']]) / models.g_mat_dict[mat]['nr_ingr']
        models.g_mat_dict[mat]['perc_pref_n'] = np.count_nonzero(bompartp * models.g_C[iX['perc_pref']]) / models.g_mat_dict[mat]['nr_ingr']
        models.g_mat_dict[mat]['perc_iff_q']  = np.nansum(bompartp * models.g_C[iX['perc_iff']]) / Q
        models.g_mat_dict[mat]['perc_pref_q'] = np.nansum(bompartp * models.g_C[iX['perc_pref']]) / Q

        models.g_M[iX['bompartp'], matix]   = Q
        models.g_M[iX['nr_ingr'], matix]    = np.sum(models.g_X[iX['nr_ingr'],matix])
        models.g_M[iX['mnc'], matix]        = np.nansum(bompartp * models.g_C[iX['mnc']]) / 1000.0
        models.g_M[iX['cut_off'], matix]    = np.nansum(np.divide(bompartp, models.g_C[iX['cut_off']]))
        models.g_M[iX['perc_iff_q'], matix]   = np.nansum(bompartp * models.g_C[iX['perc_iff']]) / Q
        models.g_M[iX['perc_pref_q'], matix]  = np.nansum(bompartp * models.g_C[iX['perc_pref']]) / Q


def calculate():
    models.g_calcmat_l = []
    models.g_calcmat_l.append(('avg_wt', '', '{0:5.2f}'.format(np.nanmean(models.g_M[iX['bompartp']]))))
    models.g_calcmat_l.append(('std_wt', '', '{0:5.2f}'.format(np.nanstd(models.g_M[iX['bompartp']]))))
    models.g_calcmat_l.append(('avg_nr', '', '{0:5.2f}'.format(np.nanmean(models.g_M[iX['nr_ingr']]))))
    models.g_calcmat_l.append(('std_nr', '', '{0:5.2f}'.format(np.nanstd(models.g_M[iX['nr_ingr']]))))
    models.g_calcmat_l.append(('min_nr', '10', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['nr_ingr']] < 10))))
    models.g_calcmat_l.append(('max_nr', '20', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['nr_ingr']] > 20))))
    models.g_calcmat_l.append(('min_price', '5', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['mnc']] < 5))))
    models.g_calcmat_l.append(('max_price', '15', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['mnc']] > 15))))
    models.g_calcmat_l.append(('min_X', '-2', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['X']] < -2))))
    models.g_calcmat_l.append(('max_X', '+2', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['X']] > 2))))
    models.g_calcmat_l.append(('min_Y', '-2', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['Y']] < -2))))
    models.g_calcmat_l.append(('max_Y', '+2', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['Y']] > 2))))
    models.g_calcmat_l.append(('min_odor', '1', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['odor']] < 1))))
    models.g_calcmat_l.append(('max_slope', '0.4', '{0:5d}'.format(np.count_nonzero(models.g_M[iX['slope']] > 0.4))))

    models.g_calcomp_l = []
    models.g_calcomp_l.append(('avg_wt', '', '{0:5.2f}'.format(np.nanmean(models.g_C[iX['bompartp']]))))
    models.g_calcomp_l.append(('std_wt', '', '{0:5.2f}'.format(np.nanstd(models.g_C[iX['bompartp']]))))
    models.g_calcomp_l.append(('avg_nr', '', '{0:5.2f}'.format(np.nanmean(models.g_C[iX['nr_ingr']]))))
    models.g_calcomp_l.append(('std_nr', '', '{0:5.2f}'.format(np.nanstd(models.g_C[iX['nr_ingr']]))))
    models.g_calcomp_l.append(('min_nr', '10', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['nr_ingr']] < 10))))
    models.g_calcomp_l.append(('max_nr', '20', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['nr_ingr']] > 20))))
    models.g_calcomp_l.append(('min_price', '5', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['mnc']] < 5))))
    models.g_calcomp_l.append(('max_price', '15', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['mnc']] > 15))))
    models.g_calcomp_l.append(('min_X', '-2', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['X']] < -2))))
    models.g_calcomp_l.append(('max_X', '+2', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['X']] > 2))))
    models.g_calcomp_l.append(('min_Y', '-2', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['Y']] < -2))))
    models.g_calcomp_l.append(('max_Y', '+2', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['Y']] > 2))))
    models.g_calcomp_l.append(('min_odor', '1', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['odor']] < 1))))
    models.g_calcomp_l.append(('max_slope', '0.4', '{0:5d}'.format(np.count_nonzero(models.g_C[iX['slope']] > 0.4))))

         
def retrieve_genesis(input_file, data_list_file, be):
    models.g_unknown_ipc = []
    retrieve(input_file, data_list_file, be)
    prepare(models.gns_e2e)
    calculate()

def retrieve_phoenix(project):
    models.g_unknown_ipc = []
    pre_e2e = predict_ml.retrieve_subm_onl (project, '', '')
    prepare(pre_e2e)
    calculate()

def draw_sample(experiment, nrcomp):
    population = np.arange(len(models.g_C[iX['nr_ingr']])) # cmpix
    probability = models.g_C[iX['nr_ingr']]
    totalcomp = np.sum(probability)
    probability = probability / totalcomp
    sample = np.random.choice(population, size=nrcomp, replace=False, p=probability)
    bom = []
    totalparts = 0.0
    for cmpix in sample:
        comp = models.g_components[cmpix]
        mu = models.g_comp_dict[comp]['avg']/1000.0
        std = models.g_comp_dict[comp]['std']
        dosage = stats.truncnorm.rvs(-mu/std, (1-mu)/std, mu, std, size=1)
        totalparts = totalparts + dosage[0]
        comp_name = models.g_data_file.ix[comp].comp_name
        bom.append([comp, comp_name, dosage[0]])
# normalize bomparts
    for ix, comp in enumerate(bom):
        bompartp = (comp[2] / totalparts) * 1000
        bom[ix] = [comp[0], comp[1], bompartp, bompartp]
    return bom


def generate_formulas(be, nrexp):
    global iX

# X is the number of components used by materials, based on normal distribution calc components for generated experiments
    nrcomp_X = models.g_M[iX['nr_ingr']]
    nrcomp_mu, nrcomp_std = stats.norm.fit(nrcomp_X)
    nrcomp_Y = stats.truncnorm.rvs(1, (1000-nrcomp_mu)/nrcomp_std,
                                   nrcomp_mu, nrcomp_std, size=nrexp+1).astype(int)
# Generate new E2E dataset for new experimentes
    gen_e2e = DataFrame(columns=['material','component','bomparts','bompartp'])
    models.fdm.generate_li = None # cleanup old data
    models.fdm.generate_li = []
    models.g_gen = None
    segment = 'Genesis:'
    for expnr in range (1, nrexp+1):
        experiment = 'FDM%0.9d' % expnr
        nrcomp = nrcomp_Y[expnr]
        models.g_gen = {'mat': experiment, 'nr_ingr': nrcomp }
        bom = draw_sample(experiment, nrcomp)
        e2eexp = pd.DataFrame(bom, columns=['component', 'comp_name', 'bomparts', 'bompartp'])
        e2eexp['material'] = experiment
        gen_e2e = gen_e2e.append(e2eexp, ignore_index=True)
        gen1 = models.generate_cl(experiment, segment, nrcomp)
        gen1.bom = bom
        models.fdm.generate_li.append(gen1)
        gen_bom = {}
        for bomline in bom:
            gen_bom[bomline[0]] = {'comp_name': bomline[1], 'bompartp': bomline[3]}
    return gen_e2e

def predict_formulas(gen_e2e):
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


def generate(be, nrexp):
    models.g_unknown_ipc_l = []
    gen_e2e = generate_formulas(be, nrexp)
    prepare(gen_e2e)
    calculate()

def predict():
    if models.fdm.learn_li != []:
        predict_formulas(gen_e2e)


    






