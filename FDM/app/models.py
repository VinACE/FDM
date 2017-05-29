"""
Definition of models.
"""

import queue
from pandas import Series, DataFrame
from django.db import models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Create your models here.


ML = {'svm'  : ["RBF SVM", SVC(gamma=2, C=1)],
      'logit': ["Logit", LogisticRegression(C=1000.0, solver='sag', random_state=0)],
      'nn'   : ["Perceptron", Perceptron(n_iter=51)],
      'bayes': ["Bayes", GaussianNB(),]}


e2e_columns_descr = {
    'fiscper'           :'Fiscal Period',
    'catclass'          :'Category',
    'enduse'            :'End Use',
    'ccprj'             :'Creative Center (Project)',
    'ccrsc'             :'Creative Center (Resource)',
    'opptyp'            :'Opportunity Type',
    'oppsts'            :'Opportunity Status',
    'prjtyp'            :'Project Type',
    'prjsts'            :'Project Status',
    'fnressts'          :'Final Result Status Project',
    'project'           :'Project',
    'region'            :'Region',
    'country_group'     :'Country-Group',
    'country'           :'Country',
    'custcat'           :'Customer Category',
    'custcorp'          :'Corporation',
    'sold-to'           :'Sold-to',
    'subm'              :'Submission',
    'material'          :'Material',
    'fnstssbm'          :'Final Result Status Subm (WIN/LOSS)',
    'experiment'        :'Experiment',
    'component'         :'Component'
    }

class learn_cl:
    segment = ''
    clf_name = ''
    def clf ():
        return 0
    y_ts_wins = 0
    y_pr_wins = 0
    y_ts_loss = 0
    y_pr_loss = 0
    accuracy = 0.0

    def __init__ (self, segment, clf_name, clf, y_ts_wins, y_pr_wins, y_ts_loss, y_pr_loss, accuracy):
        self.segment = segment
        self.clf_name = clf_name
        self.clf = clf
        self.y_ts_wins = y_ts_wins
        self.y_pr_wins = y_pr_wins
        self.y_ts_loss = y_ts_loss
        self.y_pr_loss = y_pr_loss
        self.accuracy = accuracy

class prediction_cl:
    experiment = ''
    material = ''
    descr = ''
    segment = ''
    clf_name = ''
    project = ''
    fnresstssbm = ''
    accuracy = 0.0

    def __init__ (self, experiment, descr, segment, clf_name, project,  fnresstssbm, accuracy):
        self.experiment = experiment
        self.descr = descr
        self.segment = segment
        self.clf_name = clf_name
        self.project = project
        self.fnresstssbm = fnresstssbm
        self.accuracy = accuracy

    def __init__ (self, material, descr, segment, clf_name, project,  fnresstssbm, accuracy):
        self.material = material
        self.descr = descr
        self.segment = segment
        self.clf_name = clf_name
        self.project = project
        self.fnresstssbm = fnresstssbm
        self.accuracy = accuracy

class generate_cl:
    experiment = ''
    segment = ''
    nrcomp = 0
    predictions = []
    bom = []

    def __init__ (self, experiment, segment, nrcomp):
        self.experiment = experiment
        self.segment = segment
        self.nrcomp = nrcomp
        self.predictions = []
        self.bom = []


class fdm_cl:
    comp_names = {}
    components = []
    learn_li = []
    generate_li = []
    basket_li = {}
    genesis_li = {}
    comp_dict = {}
    e2e = DataFrame()

gns_e2e = DataFrame()
g_data_list = DataFrame()
g_components = []
g_comp_dict = {}
g_unknown_ipc_l = []
g_selected = []
g_sel_form = []
g_mat_dict = {}
g_gen_dict = {}
g_calcmat_l = []
g_X = None
g_C = None
g_M = None

fdm = fdm_cl
basket_q = queue.Queue()



