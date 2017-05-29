# Read Bex queries and turn the result into JSON
# Bex query is read as an odata service
#
# It would be nice that the bex query could be directly read from BW
#

import pandas as pd
import numpy as np
import requests

def bex_query(query, **kwargs):
    urlq = query + "/A" + query
    nrvar = 0
    urlv = ""
    for varname, varvalue in kwargs.items():
        if nrvar == 0:
            urlv = "("
        else:
            urlv = urlv + ","
        urlv = urlv + varname + "='" + varvalue + "'"
        nrvar = nrvar + 1
    if nrvar > 1:
        urlv = urlv + ")"
#    url = "http://sapdgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"
#    url = "http://sapqgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"
    url = "http://sappgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"

    resp = requests.get(url, auth=('BWRA', 'sapsapif'))   
    if resp.status_code != 200:
        return([])

    data = resp.json()
    q = data['d']['results']
    df = pd.DataFrame(q)

    df = df.drop(['ISTOTAL', 'ROWID','ZSDSAM02_EQ004_BUD_F', 'ZSDSAM02_EQ004_BUDQ_F', 'ZSDSAM02_EQ004_LYR_F', 'ZSDSAM02_EQ004_LYRQ_F',
                  'ZSDSAM02_EQ004_TYR_F', 'ZSDSAM02_EQ004_TYRQ_F', 'ZSDSAM02_EQ004_OIH_F', 'ZSDSAM02_EQ004_OIHQ_F',
                  'ZSDSAM02_EQ004_ORD_F', 'ZSDSAM02_EQ004_ORDQ_F'], axis=1) 
    df.rename(columns = {'A0FISCPER3':'period', 'ZAREA_T':'region', 'ZSDSAM02_EQ004_BUD':'BUD', 'ZSDSAM02_EQ004_BUDQ':'BUDQ',
                         'ZSDSAM02_EQ004_LYR':'LYR', 'ZSDSAM02_EQ004_LYRQ':'LYRQ', 'ZSDSAM02_EQ004_TYR':'TYR', 'ZSDSAM02_EQ004_TYRQ':'TYRQ',
                         'ZSDSAM02_EQ004_OIH':'OIH', 'ZSDSAM02_EQ004_OIHQ':'OIHQ', 'ZSDSAM02_EQ004_ORD':'ORD', 'ZSDSAM02_EQ004_ORDQ':'ORDQ'}, inplace=True)

    json = df.to_json(orient='records')
    return (json)
                

def bex_ZSDE2E003_EQ003(**kwargs):
    query = 'ZSDE2E003_EQ003'
    urlq = query + "/A" + query
    nrvar = 0
    urlv = ""
    for varname, varvalue in kwargs.items():
        if nrvar == 0:
            urlv = "("
        else:
            urlv = urlv + ","
        urlv = urlv + varname + "='" + varvalue + "'"
        nrvar = nrvar + 1
    if nrvar > 1:
        urlv = urlv + ")"
#    url = "http://sapdgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"
    url = "http://sapqgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"
#    url = "http://sappgw.iff.com:8000/sap/opu/odata/sap/" + urlq + urlv + "/Results/?$format=json"

    resp = requests.get(url, auth=('BWRA', 'sapsapif'))   
    if resp.status_code != 200:
        return([])

    data = resp.json()
    q = data['d']['results']
    df = pd.DataFrame(q)

    df = df.drop(['ISTOTAL', 'ROWID','ZSDSAM02_EQ004_BUD_F', 'ZSDSAM02_EQ004_BUDQ_F', 'ZSDSAM02_EQ004_LYR_F', 'ZSDSAM02_EQ004_LYRQ_F',
                  'ZSDSAM02_EQ004_TYR_F', 'ZSDSAM02_EQ004_TYRQ_F', 'ZSDSAM02_EQ004_OIH_F', 'ZSDSAM02_EQ004_OIHQ_F',
                  'ZSDSAM02_EQ004_ORD_F', 'ZSDSAM02_EQ004_ORDQ_F'], axis=1) 
    df.rename(columns = {'A0FISCPER3':'period', 'ZAREA_T':'region', 'ZSDSAM02_EQ004_BUD':'BUD', 'ZSDSAM02_EQ004_BUDQ':'BUDQ',
                         'ZSDSAM02_EQ004_LYR':'LYR', 'ZSDSAM02_EQ004_LYRQ':'LYRQ', 'ZSDSAM02_EQ004_TYR':'TYR', 'ZSDSAM02_EQ004_TYRQ':'TYRQ',
                         'ZSDSAM02_EQ004_OIH':'OIH', 'ZSDSAM02_EQ004_OIHQ':'OIHQ', 'ZSDSAM02_EQ004_ORD':'ORD', 'ZSDSAM02_EQ004_ORDQ':'ORDQ'}, inplace=True)

    json = df.to_json(orient='records')
    return (json)
    pass
