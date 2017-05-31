
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import queue
from django.core.files import File
import glob, os
import pickle
import requests
import urllib
from itertools import combinations
import time
from collections import defaultdict
from django.http import HttpRequest
from django.http import HttpResponse
from django.http import HttpResponseRedirect
import json
import csv

import app.models as models
from FDM.settings import BASE_DIR
from app.learn_ml import *
from .basket_fp_growth import *

basket_e2e = pd.DataFrame()

def data_pass(transactions, comp_counts_s, support, pass_nbr, candidate_dct, component_set):
    for transaction in transactions:
        components = [ c for c in transaction if comp_counts_s[c] >= support ]
        candidate_dct = update_candidates(components, candidate_dct, pass_nbr, component_set)
        
    candidate_dct = clear_items(candidate_dct, support, pass_nbr)
    
    return candidate_dct



def update_candidates(item_lst, candidate_dct, pass_nbr, component_set):
    if pass_nbr==1:
        for item in item_lst:
            candidate_dct[(item,)]+=1
    else:
        frequent_items_set = set()
        for item_tuple in combinations(sorted(item_lst), pass_nbr-1):    
            if item_tuple in candidate_dct:
                frequent_items_set.update(item_tuple)
                    
        for item_set in combinations(sorted(frequent_items_set), pass_nbr):
            if len(component_set) == 0:
                candidate_dct[item_set]+=1
            elif len(component_set & set(item_set)) > 0:
                candidate_dct[item_set]+=1
        
    return candidate_dct

def clear_items(candidate_dct, support, pass_nbr):
    for item_tuple, cnt in list(candidate_dct.items()):
        if cnt<support or len(item_tuple)<pass_nbr:
            del candidate_dct[item_tuple]
    return candidate_dct


def apriori(transactions, comp_counts_s, support, itemset_size, component_set):
    candidate_dct = defaultdict(lambda: 0)
    pass_nbr = 1
    for i in range(itemset_size):
        now = time.time()
        candidate_dct = data_pass(transactions, comp_counts_s, support, pass_nbr, candidate_dct, component_set)
        pass_nbr = pass_nbr + 1
        msg = "pass number %i took %f and found %i candidates" % (pass_nbr, time.time()-now, len(candidate_dct))
        print (msg)
        models.basket_q.put(msg)
    return candidate_dct


def retrieve_basket_e2e(region_choices, category_choices, corp_choices, fnstssbm_choices):
    global basket_e2e
# Read E2E data
    basket_e2e = None
    basket_e2e = models.fdm.e2e.copy(deep=True)
    if len(region_choices):
        basket_e2e = basket_e2e[basket_e2e['region'].isin(region_choices)]
    if len(category_choices):
        basket_e2e = basket_e2e[basket_e2e['catclass'].isin(category_choices)]
    if len(corp_choices):
        basket_e2e = basket_e2e[basket_e2e['custcorp'].isin(corp_choices)]
    if len(fnstssbm_choices) < 2:
        basket_e2e = basket_e2e[basket_e2e['fnstssbm'].isin(fnstssbm_choices)]


def basket_ds(association_choice, basket_size, basket_frequency, region_choices, category_choices, corp_choices, fnstssbm_choices, component_field, material_field):
    global basket_e2e

    mat_counts_s  = basket_e2e['material'].value_counts()
    comp_counts_s  = basket_e2e['component'].value_counts()
    
    component_set = set()
    if len(material_field) > 0:
        component_set = set(basket_e2e[basket_e2e['material'] == material_field]['component'])
    if len(component_field) > 0:
        component_set.add(component_field)

    transactions = []
    for mat, count in mat_counts_s.iteritems():
        components = set(basket_e2e[basket_e2e['material'] == mat]['component'])
        if len(component_set) > 0:
            if len(component_set & components) > 0:
                transactions.append(list(components))
        else:
            transactions.append(list(components))

    if basket_frequency[-1] == '%':
        support = int(len(transactions) * (int(basket_frequency[0:-1]) / 100))
    else:
        support = int(basket_frequency)
    itemset_size = basket_size

    basket_clearresults()
    if association_choice == 'fp':
        file_name = os.path.join(BASE_DIR, 'data/fpbasket.csv')
        file = open(file_name, 'w')
        pyfile = File(file)
        #for transaction in transactions:
        #    i = 0
        #    for component in transaction:
        #        if i > 0:
        #            pyfile.write(', ')
        #        pyfile.write(component)
        #        i = i + 1
        #    pyfile.write('\n')
        #pyfile.close
        #file = open('data/tsk.csv', 'r')
        #pyfile = File(file)
        #transactions = []
        #for line in pyfile:
        #    transaction =line.rstrip().split(',')
        #    transactions.append(transaction)
        #pyfile.close
        models.basket_q.put("Starting basket search using FP Growth")
        itemsets_dct = basket_fp_growth(transactions, support, itemset_size)
    elif association_choice == 'apriori':
        models.basket_q.put("Starting basket search using Apriori")
        itemsets_dct = apriori(transactions, comp_counts_s, support, itemset_size, component_set)

    print(dict(list(itemsets_dct.items())[0:10]))
    basket_li = []
    for itemset in itemsets_dct.items():
        basket = []
        for comp in itemset[0]:
            if comp in models.fdm.comp_names:
                comp_name = models.fdm.comp_names[comp]
            else:
                comp_name = "#"
            basket.append([comp, comp_name])
        freq = itemset[1]
        support = freq / len(transactions)
        size = len(itemset[0])
        basket_li.append([basket, freq, support, size])
#   basket_li = list(itemsets_dct.items())
    return basket_li

def basket_save(region_choices, category_choices, corp_choices):
    choices = ""
    for choice in region_choices + category_choices + corp_choices:
        choices = choices + choice + '_'
    ml_file = os.path.join(BASE_DIR, 'data/' + choices + 'basket.pickle') 
    try:
        file = open(ml_file, 'wb')
        pyfile = File(file)
        pickle.dump(models.fdm.basket_li, pyfile, protocol=pickle.HIGHEST_PROTOCOL)
        pyfile.close()
        return True
    except:
        return False

def basket_retrieve(region_choices, category_choices, corp_choices):
    choices = ""
    for choice in region_choices + category_choices + corp_choices:
        choices = choices + choice + '_'
    ml_file = os.path.join(BASE_DIR, 'data/' + choices + 'basket.pickle') 
    try:
        file = open(ml_file, 'rb')
        pyfile = File(file)
        models.fdm.basket_li = pickle.load(pyfile)
        pyfile.close()
        return True
    except:
        return False


def basket_experiment(basket_id):
    global basket_e2e
    basket_id = int(basket_id)
    basket = models.fdm.basket_li[basket_id][0]
    basket_size = len(basket)
    experiments = set()
    components = [comp[0] for comp in basket]
    for comp in components:
        new_experiments = set(basket_e2e[basket_e2e['component']==comp]['material'])
        if len(experiments) == 0:
            experiments = new_experiments
        else:
            experiments = experiments & new_experiments
    experiments = list(experiments)
    basket_experiment_li = []
    for exp in experiments:
        e2e_row = basket_e2e[basket_e2e['material'] == exp].iloc[0]
        fnstssbm = e2e_row['fnstssbm']
        bom = list(set(basket_e2e[basket_e2e['material'] == exp]['component']))
        if len(bom) > basket_size:
            basket_experiment_li.append([exp, fnstssbm, bom])
    return basket_experiment_li


def radial_tree_experiments(basket_experiment_li):
    children = []
    for exp in basket_experiment_li:
        children.append({"name":exp[0], "size":3000})
    return children


def basket_clearresults():
    while not models.basket_q.empty():
        try:
            models.basket_q.get(False)
        except Empty:
            continue


def basket_pollresults_api(request):
    try:
        msg = models.basket_q.get(block=True, timeout=10)
    except queue.Empty:
        msg = "No update, still working..."
    msg_json = json.dumps(msg)
    return HttpResponse(msg_json, content_type='application/json')


def basket_experiment_api(request, basket_id):
    global basket_e2e
    df = pd.DataFrame(columns=['material','sold-to','fnstssbm'])
    rownr = 0
    basket_id = int(basket_id)
    basket = models.fdm.basket_li[basket_id][0]
    basket_size = len(basket)
    experiments = set()
    components = [comp[0] for comp in basket]
    for comp in components:
        new_experiments = set(basket_e2e[basket_e2e['component'] == comp]['material'])
        if len(experiments) == 0:
            experiments = new_experiments
        else:
            experiments = experiments & new_experiments
    experiments = list(experiments)
    basket_experiment_li = []
    for exp in experiments:
        e2e_row = basket_e2e[basket_e2e['material'] == exp].iloc[0]
        fnstssbm = e2e_row['fnstssbm']
        sold_to = e2e_row['sold-to']
        bom = list(basket_e2e[basket_e2e['material'] == exp]['component'])
        if len(bom) > basket_size:
            basket_experiment_li.append([exp, fnstssbm, bom])
        df.loc[rownr] = {'material': exp, 'sold-to': sold_to, 'fnstssbm': fnstssbm  }
        rownr = rownr + 1

    tree1 = {"name": "basket", "children" : radial_tree_experiments(basket_experiment_li)}
    tree = {"name": "basket", "children" : radial_tree(df, 'sold-to', ['fnstssbm', 'material'])}
    tree_json = json.dumps(tree)

    return HttpResponse(tree_json, content_type='application/json')

def radial_tree(df, level, child_levels):
    children = []
    level_values = df[level].unique()
    for lv in level_values:
        if len(child_levels) == 0:
            children.append({"name": lv, "size": 3000})
        else:
            df2 = df[df[level] == lv]
            children.append({"name": lv, "children": radial_tree(df2, child_levels[0], child_levels[1:])})
    return children
       

    






