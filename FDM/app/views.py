"""
Definition of views.
"""

from pandas import Series, DataFrame
from django.shortcuts import render, render_to_response, redirect
from django.template.context_processors import csrf
from django.http import HttpRequest
from django.http import HttpResponse
from django.http import HttpResponseRedirect
#from django.http import JsonResponse
from django.template import RequestContext
from django.core.files import File
from datetime import datetime
import glob, os
import pickle
import json
import app.models as models
from FDM.settings import BASE_DIR
from .forms import *
from .retrieve_ds import *
from .learn_ml import *
from .predict_ml import *
from .generate_ml import *
from .basket_ds import *
from .explore_ds import *
import app.genesis as genesis


def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(request, 'app/index.html', {'title':'Home Page', 'year':datetime.now().year})


def retrieve_view(request):
    """Renders the retrieve page."""
    form_errors = []
    if request.method == 'POST':
        e2e_prediction = request.POST['e2e_choices']
        retrieve_ml(e2e_prediction)
        retrieve_df = models.fdm.e2e.describe(include=['object']).transpose()
        retrieve_df.rename(index = models.e2e_columns_descr, inplace=True)
        return render(request, 'app/retrieveresults.html',
                            {'retrieve_df': retrieve_df.to_html(index=True, classes="table table-striped") })

    return render(request, 'app/retrieve.html', {'form_errors': form_errors, 'message':'IFF-DM Data Miner', 'year':datetime.now().year} )

def learn_view(request):
    """Renders the learn page."""
    if request.method == 'POST':
        form = learn_form(request.POST)
        if form.is_valid():
            model_choice = form.cleaned_data['model_choice_field']
            if model_choice == 'new':
                model_name = form.cleaned_data['model_name_field']
            else:
                model_name = model_choice
            ml_choices = form.cleaned_data['ml_choices_field']
            ms_choices = form.cleaned_data['ms_choices_field']
            test_perc = form.cleaned_data['test_perc_field']
            win_weight = form.cleaned_data['win_weight_field']
#            learn1 = models.learn_cl
#            learn1.segment = ['10', '20']
#            models.fdm.learn_li = []
#            models.fdm.learn_li.append(learn1)
#            models.fdm.components = ['123', '456']
#            file = open(ml_file, 'wb')
#            myfile = File(file)
#            try:
#                pickle.dump(models.fdm.learn_li, myfile)
#            except PicklingError:
#                print('PicklingError')
#            myfile.close()
#            ml_file = ml_file + '2'
#            file = open(ml_file, 'wb')
#            myfile = File(file)
#            try:
#                pickle.dump(models.fdm.components, myfile)
#            except PicklingError:
#                print('PicklingError')
#            myfile.close()
#
#            ml_file = form.data['ml_file_field']
#            models.fdm.learn_li = []
#            models.fdm.components.clear()
#            file = open(ml_file, 'rb')
#            myfile = File(file)
#            models.fdm.learn_li = pickle.load(myfile)
#            myfile.close()
#            ml_file = ml_file + '2'
#            file = open(ml_file, 'rb')
#            myfile = File(file)
#            models.fdm.components = pickle.load(myfile)
#            myfile.close()

            comp_file = os.path.join(BASE_DIR, 'data/components.csv')
            if os.path.exists(comp_file) and len(models.fdm.comp_names) == 0:
                read_components()
            if 'learn' in form.data:
                learn_ml(ml_choices, ms_choices, test_perc, win_weight)
                if len(model_name) > 0:
                    ml_file = os.path.join(BASE_DIR, 'data/' + model_name + '_1.pickle')
                    file = open(ml_file, 'wb')
                    pyfile = File(file)
                    pickle.dump(models.fdm.learn_li, pyfile)
                    pyfile.close()
                    ml_file = os.path.join(BASE_DIR, 'data/' + model_name + '_2.pickle')
                    file = open(ml_file, 'wb')
                    pyfile = File(file)
                    pickle.dump(models.fdm.comp_dict, pyfile)
                    pyfile.close()
                return render(request, 'app/learnresults.html', {'learn_li' : models.fdm.learn_li})
            elif 'retrieve' in form.data:
                if len(model_name) > 0:
                    models.fdm.learn_li = []
                    models.fdm.components = []
                    ml_file = os.path.join(BASE_DIR, 'data/' + model_name + '_1.pickle')
                    file = open(ml_file, 'rb')
                    pyfile = File(file)
                    models.fdm.learn_li = pickle.load(pyfile)
                    pyfile.close
                    ml_file = os.path.join(BASE_DIR, 'data/' + model_name + '_2.pickle')
                    file = open(ml_file, 'rb')
                    pyfile = File(file)
                    models.fdm.comp_dict = pickle.load(pyfile)
                    pyfile.close                                                           
                return render(request, 'app/learnresults.html', {'learn_li' : models.fdm.learn_li})
        else:
            return render(request, 'app/learn.html', {'form': form } )
    else:
        form = learn_form(initial={'classification_field':'winloss', 'e2e_prediction_field':'e2esubm',
                                   'ml_choices_field':['svm','logit','nn', 'bayes'], 'ms_choices_field':['region']})
        if models.fdm.e2e.empty:
            form.add_form_error("First retrieve your E2E data")

    return render(request, 'app/learn.html', {'form': form, 'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )


def predict_view(request):
    """Renders the predict page."""
    if request.method == 'POST':
        form = predict_form(request.POST)
        if form.is_valid():
            e2e_prediction = form.cleaned_data['e2e_prediction_field']
            project = form.cleaned_data['project_field']
            experiment = form.cleaned_data['experiment_field']
            submission = form.cleaned_data['submission_field']
            prediction_li = predict_ml(e2e_prediction, project, experiment, submission)
            df = predict_df (prediction_li)
            return render(request, 'app/predictresults.html',
                            {'project': project, 'prediction_df': df.to_html(index=False, classes="table table-striped") })
    else:
        form = predict_form(initial={'e2e_prediction_field':'e2esubm_onl'})
        if models.fdm.e2e.empty:
            form.add_form_error("First retrieve your E2E data")
        if len(models.fdm.learn_li) == 0:
            form.add_form_error("First learn a ML model")

    return render(request, 'app/predict.html', {'form': form, 'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )


def generate_view(request):
    """Renders the generate page."""
    if request.method == 'POST':
        form = generate_form(request.POST)
        if form.is_valid():
            project = form.cleaned_data['project_field']
            mnc = form.cleaned_data['mnc_field']
            region_choices = form.cleaned_data['region_choices_field']
            category_choices = form.cleaned_data['category_choices_field']
            corp_choices = form.cleaned_data['corp_choices_field']
            nrexp = form.cleaned_data['nrexp_field']
            if 'generate' in form.data:
                generate_ml(nrexp, region_choices, category_choices, corp_choices)
            prediction_li = predict_gen_ml()
            df = predict_df (prediction_li)
            return render(request, 'app/generateresults.html', {'generate_li' : models.fdm.generate_li,
                                                                'prediction_df': df.to_html(index=False, classes="table table-striped") })
    else:       
        form = generate_form()
        if models.fdm.e2e.empty:
            form.add_form_error("First retrieve your E2E data")
        if len(models.fdm.learn_li) == 0:
            form.add_form_error("First learn a ML model")

    return render(request, 'app/generate.html', {'form': form, 'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )


def generate_experiment_view(request, experiment_id):
    """Renders the experiment page."""
    assert isinstance(request, HttpRequest)
    experiment = request.GET.get('experiment_id', '')
    gen1 = [ exp for exp in models.fdm.generate_li if exp.experiment == experiment_id]
    bom = []
    if len(gen1) > 0:
        bom = gen1[0].bom
    return render(request, 'app/generate_experiment.html', {'experiment' : experiment_id, 'bom' : bom})

def explore_view(request):
    """Renders the explore page."""
    return render(request, 'app/explore.html', {'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )

def basket_view(request):
    """Renders the basket page."""
    if request.method == 'POST':
        form = basket_form(request.POST)
        if form.is_valid():
            region_choices = form.cleaned_data['region_choices_field']
            category_choices = form.cleaned_data['category_choices_field']
            corp_choices = form.cleaned_data['corp_choices_field']
            fnstssbm_choices = form.cleaned_data['fnstssbm_choices_field']
            component_field = form.cleaned_data['component_field']
            material_field = form.cleaned_data['material_field']
            association_choice = form.cleaned_data['association_choice_field']
            basket_size = form.cleaned_data['basket_size_field']
            basket_frequency = form.cleaned_data['basket_frequency_field']
            retrieve_basket_e2e(region_choices, category_choices, corp_choices, fnstssbm_choices)
            if 'generate' in form.data:
                models.fdm.basket_li = None
                models.fdm.basket_li = basket_ds(association_choice, basket_size, basket_frequency,
                                                 region_choices, category_choices, corp_choices, fnstssbm_choices,
                                                 component_field, material_field)
                if not basket_save(region_choices, category_choices, corp_choices):
                    form.add_form_error("Could not save basket results")
            if 'retrieve' in form.data:
                if not basket_retrieve(region_choices, category_choices, corp_choices):
                    form.add_form_error("Could not retrieve basket results")
            if len(form._errors) == 0:
                return render(request, 'app/basketresults.html', {'basket_li' : models.fdm.basket_li } )
    else:
        form = basket_form(initial={'association_choice_field':'apriori', 'fnstssbm_choices_field':['WIN','LOSS']})
        if models.fdm.e2e.empty:
            form.add_form_error("First retrieve your E2E data")
    return render(request, 'app/basket.html', {'form': form, 'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )


def basket_experiment_view(request, basket_id):
    """Renders the experiment page."""
    assert isinstance(request, HttpRequest)
    basket_id = int(basket_id)
    basket = models.fdm.basket_li[basket_id][0]
    basket_experiment_li = basket_experiment(basket_id)
    return render(request, 'app/basket_experiment.html', {'basket': basket, 'basket_experiment_li' : basket_experiment_li })


def explore_compfreq_view(request):
    """Renders the explore page."""
    return render(request, 'app/explore/explore_compfreq.html',
                  {'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )

def explore_e2esubm_view(request):
    """Renders the explore page."""
    return render(request, 'app/explore/explore_e2esubm.html',
                  {'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )

def explore_orderbook_view(request):
    """Renders the explore page."""
    return render(request, 'app/explore/orderbook.html',
                  {'message':'IFF-DM Data Miner', 'year':datetime.now().year,} )


def genesis_view(request):
    """Renders the genesis page."""
    if request.method == 'POST':
        form = genesis_form(request.POST)
        if 'set_selection' in form.data:
            models.g_selected = request.POST.getlist('selected[]')
            models.g_sel_form = request.POST.getlist('sel_form[]')
            return redirect('genesis')
        if 'return_selection' in form.data:
            return redirect('genesis')
        if any([x in form.data for x in ['predict','generate', 'retrieve_phoenix']]) and len(models.g_comp_dict) == 0:
            form.add_form_error("First retrieve Genesis data")
        if form.is_valid():
            input_file = form.cleaned_data['input_file_field']
            data_list_file = form.cleaned_data['data_list_file_field']
            be = form.cleaned_data['be_field']
            project = form.cleaned_data['project_field']
            nrexp = form.cleaned_data['nrexp_field']
            if 'retrieve_genesis' in form.data:
                genesis.retrieve_genesis(input_file, data_list_file, be)
                g_mode = 'retrieve'
            if 'retrieve_phoenix' in form.data:
                genesis.retrieve_phoenix(project)
                g_mode = 'retrieve'
            if 'prepare_genesis' in form.data:
                genesis.retrieve_genesis(input_file, data_list_file, be)
                g_mode = 'prepare'
            if 'generate' in form.data:
                genesis.generate(be, nrexp)
                g_mode = 'generate'
            if 'predict' in form.data:
                genesis.predict()
                prediction_li = predict_gen_ml()
                df = predict_df (prediction_li)
                return render(request, 'app/generateresults.html', {'generate_li' : models.fdm.generate_li,
                                                                'prediction_df': df.to_html(index=False, classes="table table-striped") })
            if len(form._errors) == 0:
                context = {
                    'g_mode'      : g_mode,
                    'g_comp_dict' : models.g_comp_dict,
                    'g_mat_dict'  : models.g_mat_dict,
                    'g_calcmat_l' : models.g_calcmat_l,
                    'g_calcomp_l' : models.g_calcomp_l,
                    'g_unknown_ipc_l' : models.g_unknown_ipc_l,
                    'g_X'         : models.g_X,
                    }
                return render(request, 'app/genesisresults.html', context)
    else:
        form = genesis_form(initial={'input_file_field' : 'genesis_input.csv', 'data_list_file_field' : 'genesis_data_list.csv'})

    if len(models.g_selected) == 0:
        nr_sel_ingr = 'ALL'
        sel_ingr = []
    else:
        nr_sel_ingr = '{0:d}'.format(len(models.g_selected))
        sel_ingr = [(comp, models.g_comp_dict[comp]['comp_name']) for comp in models.g_selected]
    if len(models.g_sel_form) == 0:
        nr_sel_form = 'ALL'
    else:
        nr_sel_form = '{0:d}'.format(len(models.g_sel_form))
    context = {
        'form'          : form,
        'nr_sel_ingr'   : nr_sel_ingr,
        'sel_ingr'      : sel_ingr,
        'nr_sel_form'   : nr_sel_form,
        'sel_form'      : models.g_sel_form,
        }
    return render(request, 'app/genesis.html', context)
                  


def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'FDM Fragrance Data Miner',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'IFF-DM Data Miner',
            'year':datetime.now().year,
        }
    )


def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('registration/registrer_complete')

    else:
        form = RegistrationForm()
    token = {}
    token.update(csrf(request))
    token['form'] = form

    return render_to_response('registration/register.html', token)

def registrer_complete(request):
    return render_to_response('registration/registrer_complete.html')