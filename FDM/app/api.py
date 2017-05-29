"""
Definition of api-views.
"""

from pandas import Series, DataFrame
from django.shortcuts import render
from django.http import HttpRequest
from django.http import HttpResponse
from django.http import HttpResponseRedirect
#from django.http import JsonResponse
from django.template import RequestContext
from django.core.files import File
from datetime import datetime
import json
import app.models as models
from .forms import *
from .learn_ml import *
from .predict_ml import *
from .generate_ml import *
from .basket_ds import *
from .explore_ds import *
from .bex import *


def e2equery_compfreq_api(request):
    """Return query results"""
    winners = [
        {'name': 'Albert Einstein', 'category':'Physics'},
        {'name': 'V.S. Naipaul', 'category':'Literature'},
        {'name': 'Dorothy Hodgkin', 'category':'Chemistry'}]
    winners_json = json.dumps(winners)
    components_json = explore_compfreq_ml()
    return HttpResponse(components_json, content_type='application/json')
#    return render(request, winners_json)
#    return render(request, 'app/exploreresults.html', {'query' :  winners_json})
#   return JsonResponse(winners_json)

def e2equery_e2esubm_api(request):
    components_json = explore_e2esubm_ml()
    return HttpResponse(components_json, content_type='application/json')

def e2equery_orderbook_api(request):
#    json = bex_ZSDE2E003_EQ003(ZVOVSALR='5003435', VOSOLDTO='', VOMAT='')
    json = bex_query('ZSDSAM02_EQ004', ZVOPRCTR='', ZVORGN='', ZVOCCL='', ZVOCORP='', ZVOSBRGN='', ZVOCGRP='', ZVOREP='')
    return HttpResponse(json, content_type='application/json')






