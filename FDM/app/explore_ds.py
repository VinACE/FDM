
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import requests
import urllib
import json
from itertools import combinations
import time
from collections import defaultdict

import app.models as models
from app.learn_ml import *

explore_e2e = pd.DataFrame()


def explore_compfreq_ml():
    global explore_e2e

# Read E2E data
    if models.fdm.e2e.empty == True:
        models.fdm.e2e = read_subm_e2e()

    explore_e2e = models.fdm.e2e
    comp_counts_s  = explore_e2e['component'].value_counts()
    mean = comp_counts_s.mean()
    comp_counts_s = comp_counts_s[comp_counts_s > 3*mean]
    comp_counts_l = [{"comp": i, "freq": str(j)} for i,j in comp_counts_s.items()]
    components_json = json.dumps(comp_counts_l)
#   explore_e2e_json = explore_e2e.to_json()

#    comp_count_l = [
#        {'comp': 'Albert Einstein', 'freq':100},
#        {'comp': 'V.S. Naipaul', 'freq':50},
#        {'comp': 'Jan de Jager', 'freq':180},
#        {'comp': 'Dorothy Hodgkin', 'freq':120}]
#    components_json = json.dumps(comp_count_l)

    return components_json

def explore_e2esubm_ml():
    global explore_e2e

# Read E2E data
    if models.fdm.e2e.empty == True:
        models.fdm.e2e = read_subm_e2e()

    explore_e2e = models.fdm.e2e

#    e2e.columns = ('fiscper', 'catclass', 'enduse', 'prodhier', 'bomlvl', 'expcmp',
#                      'exptyp', 'co-area', 'profit-ctr', 'fiscvarnt', 'ccprj', 'ccrsc', 
#                     'opptyp', 'oppsts', 'prjown', 'prjtyp', 'prjsts', 'fnressts',
#                     'project', 'region', 'country_group', 'country', 
#                     'custcat', 'custcorp', 'sold-to', 'npidoc', 'npityp', 'subm',
#                     'material', 'fnstssbm', 'experiment', 'component', 
#                     'adoptions', 'projects', 'submissions', 'annpotval',
#                     'annpotvol', 'bomparts', 'bompartp', 'items', 'base_uom', 
#                     'doc_currcy', 'annpot_currcy', 'annpot_uom')
    e2e_sub_df = explore_e2e.groupby(['component','catclass','fiscper','region','custcorp']).size().reset_index().rename(columns={0:'count'})
    e2e_json = e2e_sub_df.to_json(orient='records')

    return e2e_json


    






