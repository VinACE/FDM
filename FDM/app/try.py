from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import scipy as sp
from datetime import datetime
from scipy import stats
import os
import glob
import json

#dataframe pivot
components=['ca', 'cb', 'cc']
pdf = pd.DataFrame([['ma', 'ca', 'LOSS', 0.01], ['mb', 'ca', 'LOSS', 0.02], ['mb', 'cb', 'LOSS', 0.03]], columns=['m', 'c', 'sts', 'partp'])
pdf2 = pdf.pivot_table(['partp'], index=['m'], columns=['c'], fill_value=0)
pdd = pdf.set_index('m')['sts'].to_dict()
ps = Series(pdd)
pdf2['sts'] = ps

#Hierarchical indexing
hdf = pd.DataFrame(np.arange(12).reshape((4,3)),
                   index=[['a', 'a', 'b', 'b'],[1, 2, 1, 2]],
                   columns=['Ohio','NJ', 'Colorado'])
hdf.ix['a']
hdf.ix['a',1]
hdf.ix['a',1].Ohio
hdf.ix['a',1]['Ohio']
hdf.set_value(('a',1),'Ohio', 88)
# insert
hdf.set_value(('a',1),'Ohio', 88)

df = pd.DataFrame([10.0, 20.0, 30.0, 40.0])
df = df[0].map('{:.0f}'.format)

df2 = pd.DataFrame(['10', '20', '30', '40'], columns=['A'], index=['a', 'b', 'c', 'd'])
df3 = df2[df2['A'].isin(['10'])]

segment_df = pd.DataFrame([['LOSS', 'LOSS', 'WIN'], [1, 2, 3]], index=['fnressts', 'count'])
segment_df = segment_df.T
nrloss = sum(segment_df[segment_df['fnressts'] == 'LOSS']['count'])
nrocc = sum(segment_df['fnressts'] == 'LOSS')

l = ['ab', 'ac', 'ad']
dict_l = dict([(e, {'ix' : l.index(e)}) for e in l])

nd2 = np.array(['ab', 'bc'], dtype=object)

s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s2 = Series(value for key, value in s.items() if value > 2 )

dict_d = {('a', 'b'): 20, ('b','c'):30}
dict_l = list(dict_d.items())

# rename columns
colmap = {'m': 'mat', 'c': 'comp', 'sts':'status', 'partp':'bompartp'}
newcol = pdf.columns.map(lambda col: colmap[col])

pdf.rename(columns = {'m':'mat'}, inplace=True)

# os
files = os.listdir("data")
tuples = [(f, f) for f in files]

winners = [
     {'name': 'Albert Einstein', 'category':'Physics'},
     {'name': 'V.S. Naipaul', 'category':'Literature'},
     {'name': 'Dorothy Hodgkin', 'category':'Chemistry'}]

winners_json = json.dumps(winners)

# Radial Reingold–Tilford Tree
#
graph = {
 "name": "flare",
 "children": [
  {
   "name": "analytics",
   "children": [
    {
     "name": "cluster",
     "children": [
      {"name": "AgglomerativeCluster", "size": 3938},
      {"name": "CommunityStructure", "size": 3812},
      {"name": "HierarchicalCluster", "size": 6714},
      {"name": "MergeEdge", "size": 743}
     ]
    }]}]}
    



datetime.strptime('20-Nov-2002','%d-%b-%Y').strftime('%Y%m%d')
datetime.strptime('Oct 04 2002','%b %d %Y').strftime('%Y%m%d')


# The rules for translating a Unicode string into a sequence of bytes are called an encoding
# in Python 3.x a string contains Unicode, UTF-8
# decoding is from bytes to Unicode string
str = "hello"
bytstr = str.encode('utf-8', 'ignore')
# unistr = unicode (str, "utf-8")

import nltk



