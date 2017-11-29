"""
Definition of urls for FDM.
"""

from datetime import datetime
from django.conf.urls import url
from django.contrib import admin
from django.contrib.auth import views as auth_views
from app.forms import BootstrapAuthenticationForm
import app.views
import app.api
import app.basket_ds

# Uncomment the next lines to enable the admin:
# from django.conf.urls import include
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = [
    # Examples:
    url(r'^$', app.views.home, name='home'),

    url(r'^retrieve$', app.views.retrieve_view, name='retrieve'),
    url(r'^learn$', app.views.learn_view, name='learn'),
    url(r'^predict$', app.views.predict_view, name='predict'),

    url(r'^generate$', app.views.generate_view, name='generate'),
    url(r'^experiment/(?P<experiment_id>[A-Z0-9]+)/$', app.views.generate_experiment_view, name='experiment'),

    url(r'^basket$', app.views.basket_view, name='basket'),
    url(r'^api/basket_pollresults$', app.basket_ds.basket_pollresults_api, name='api/basket_pollresults'),
    url(r'^basket_experiment/(?P<basket_id>[A-Z0-9]+)/$', app.views.basket_experiment_view, name='basket_experiment'),
    url(r'^api/basket_experiment_api/(?P<basket_id>[A-Z0-9]+)/$', app.basket_ds.basket_experiment_api, name='api/basket_experiment_api'),

    url(r'^explore$', app.views.explore_view, name='explore'),
    url(r'^explore/explore_compfreq$', app.views.explore_compfreq_view, name='explore/explore_compfreq'),
    url(r'^explore/explore_e2esubm$', app.views.explore_e2esubm_view, name='explore/explore_e2esubm'),
    url(r'^explore/orderbook$', app.views.explore_orderbook_view, name='explore/orderbook'),

    url(r'^genesis$', app.views.genesis_view, name='genesis'),

    url(r'^api/e2equery_compfreq$', app.api.e2equery_compfreq_api, name='api/e2equery_compfreq'),
    url(r'^api/e2equery_e2esubm$', app.api.e2equery_e2esubm_api, name='api/e2equery_e2esubm'),
    url(r'^api/e2equery_orderbook$', app.api.e2equery_orderbook_api, name='api/e2equery_orderbook'),

    url(r'^contact$', app.views.contact, name='contact'),
    url(r'^about', app.views.about, name='about'),

    # Registration URLs
    url(r'^accounts/register/$', app.views.register, name='register'),
    url(r'^accounts/register_complete/$', app.views.registrer_complete, name='register_complete'),
    url(r'^login/$', auth_views.login, name='login'),
    url(r'^logout/$', auth_views.logout, name='logout'),
    url(r'^admin/', admin.site.urls),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
    ]
