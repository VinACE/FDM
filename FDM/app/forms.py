"""
Definition of forms.
"""

import glob, os

from django import forms
from django.forms.utils import ErrorList
from django.forms.utils import ErrorDict
from django.forms.forms import NON_FIELD_ERRORS
from django.contrib.auth.forms import AuthenticationForm
from django.utils.translation import ugettext_lazy as _

from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

import app.models as models
from FDM.settings import BASE_DIR

class BootstrapAuthenticationForm(AuthenticationForm):
    """Authentication form which uses boostrap CSS."""
    username = forms.CharField(max_length=254,
                               widget=forms.TextInput({
                                   'class': 'form-control',
                                   'placeholder': 'User name'}))
    password = forms.CharField(label=_("Password"),
                               widget=forms.PasswordInput({
                                   'class': 'form-control',
                                   'placeholder':'Password'}))

def model_choices():
    file_name = os.path.join(BASE_DIR, 'data/*_1.pickle')
    files = ['new']
    glob_list = glob.glob(file_name)
    for ml_name in glob_list:
        head, tail = os.path.split(ml_name)
        files.append(tail)
    for i in range(1, len(files)):
        files[i] = files[i][0:-9]
    choices = [(file, file) for file in files]
    return (choices)

class learn_form(forms.Form):
# ml_choices have to have the same key as models.ML
    classification_choices = (('winloss', 'Win or Loss'), ('other', 'Other (TBD)'))
    classification_field = forms.ChoiceField(label='Classification', choices=classification_choices, widget=forms.RadioSelect)
    ml_choices = (('svm', 'SVM Support Vector Machine'), ('logit', 'Logit Regression'), ('nn', 'Neural Network (Perceptor)'), ('bayes', 'Naive Bayes'))
    ml_choices_field = forms.MultipleChoiceField(label='Machine Learning', choices=ml_choices, widget=forms.CheckboxSelectMultiple, required=True)
# ms choices need to have the same key as the e2e dataframe columns
    ms_choices = (('region', 'Region'), ('catclass', 'Category'), ('custcorp', 'Corporation'))
    ms_choices_field = forms.MultipleChoiceField(label='Market Segments', choices=ms_choices, widget=forms.CheckboxSelectMultiple, required=True)
    test_perc_field = forms.FloatField(label='Testing Set %', required=False, initial=30.0,
                                        help_text='Size of the testing set [%]')
    win_weight_field = forms.FloatField(label='Win Weight', required=False, initial=0.0,
                                        help_text='0 - pro-ratio, 1 - equal, x - win-weight')
    model_choice_field = forms.ChoiceField(label='FDM Model Names', choices=model_choices(), required=False)
    model_name_field = forms.CharField(label='New Model Name', max_length=30, required=False, help_text='Model parametere will be stored under this name')
    
    def clean(self):
        if models.fdm.e2e.empty:
            self.add_error(None, "First retrieve your E2E data")
    def add_form_error(self, message):
        if not self._errors:
            self._errors = ErrorDict()
        if not NON_FIELD_ERRORS in self._errors:
            self._errors[NON_FIELD_ERRORS] = self.error_class()
        self._errors[NON_FIELD_ERRORS].append(message)


class predict_form(forms.Form):
    e2e_choices = (('e2eexp_onl', 'Experiment (ONline)'), ('e2esubm_onl', 'Submission (ONline)'), ('e2eexp_ofl', 'Experiment (OFFline)'), ('e2esubm_ofl', 'Submission (OFFline)'))
    e2e_prediction_field = forms.ChoiceField(label='E2E Prediction', choices=e2e_choices, widget=forms.RadioSelect)
    project_field = forms.CharField(
        label='Project',
        max_length=30,
        required = False,
        initial = '869952WRA',
        help_text='863880WRA/875303WRA for subm, 301677WRR for exp')
    experiment_field = forms.CharField(
        label='Experiment',
        max_length=30,
        required = False,
        initial = '',
        help_text='Q4ACCO1009AO A Phoenix Experiment')
    submission_field = forms.CharField(
        label='Submission',
        max_length=30,
        required = False,
        initial = '',
        help_text='30918279 A Mona List Submission')
    def clean(self):
        if len(models.fdm.learn_li) == 0:
            self.add_error(None, "First learn a ML model")
    def add_form_error(self, message):
        if not self._errors:
            self._errors = ErrorDict()
        if not NON_FIELD_ERRORS in self._errors:
            self._errors[NON_FIELD_ERRORS] = self.error_class()
        self._errors[NON_FIELD_ERRORS].append(message)

class generate_form(forms.Form):
    project_field = forms.CharField(label='Project', max_length=30, required = False, help_text='The project for which to generate new Formulas')
    mnc_field = forms.FloatField(label='MNC Cost', required = False, help_text='Max MNC Cost')
    cc_field = forms.CharField(label='Creative Center', max_length=30, required = False, help_text='Creative Center where the formulas will be made')
    region_choices = (('10', 'North America'), ('20', 'Latin America'), ('30', 'Greater Asia'), ('40', 'Europe / A.M.E.E.'))
    region_choices_field = forms.MultipleChoiceField(label='Region', choices=region_choices, widget=forms.CheckboxSelectMultiple, required=False)
    category_choices = (('001', 'Fine Fragr'), ('002', 'Toiletries'), ('003', 'Fabric Care'), ('004', 'Home care'), ('005', 'Personal Wash'))
    category_choices_field = forms.MultipleChoiceField(label='Category', choices=category_choices, widget=forms.CheckboxSelectMultiple, required=False)
    corp_choices = (('1000740', 'Unilever'), ('1000615', 'Procter & Gamble'), ('1000275', "L'Oreal"), ('1000250', 'Colgate'), ('1000135', 'Avon'), ('1000435', 'Henkel'))
    corp_choices_field = forms.MultipleChoiceField(label='Corporation', choices=corp_choices, widget=forms.CheckboxSelectMultiple, required=False)
    nrexp_field = forms.IntegerField(label='Number of Experiment to Generate', initial = 10)
    def clean(self):
        if models.fdm.e2e.empty:
            self.add_error(None, "First retrieve your E2E data")
    def add_form_error(self, message):
        if not self._errors:
            self._errors = ErrorDict()
        if not NON_FIELD_ERRORS in self._errors:
            self._errors[NON_FIELD_ERRORS] = self.error_class()
        self._errors[NON_FIELD_ERRORS].append(message)

class basket_form(forms.Form):
    region_choices = (('10', 'North America'), ('20', 'Latin America'), ('30', 'Greater Asia'), ('40', 'Europe / A.M.E.E.'))
    region_choices_field = forms.MultipleChoiceField(label='Region', choices=region_choices, widget=forms.CheckboxSelectMultiple, required=True)
    category_choices = (('001', 'Fine Fragr'), ('002', 'Toiletries'), ('003', 'Fabric Care'), ('004', 'Home care'), ('005', 'Personal Wash'))
    category_choices_field = forms.MultipleChoiceField(label='Category', choices=category_choices, widget=forms.CheckboxSelectMultiple, required=False)
    corp_choices = (('1000740', 'Unilever'), ('1000615', 'Procter & Gamble'), ('1000275', "L'Oreal"), ('1000250', 'Colgate'), ('1000135', 'Avon'), ('1000435', 'Henkel'))
    corp_choices_field = forms.MultipleChoiceField(label='Corporation', choices=corp_choices, widget=forms.CheckboxSelectMultiple, required=False)
    fnstssbm_choices = (('WIN', 'Win'), ('LOSS', 'Loss'))
    fnstssbm_choices_field = forms.MultipleChoiceField(label='Result Status', choices=fnstssbm_choices, widget=forms.CheckboxSelectMultiple, required=False)
    association_choices = (('apriori', 'Apriori'), ('fp', 'FP-Growth'))
    association_choice_field = forms.ChoiceField(label='Association Logic', choices=association_choices, widget=forms.RadioSelect, required=True)
    component_field = forms.CharField(label='Component', max_length=8, required = False, initial = '', help_text='basket containing this component')
    material_field = forms.CharField(label='Material', max_length=8, required = False, initial = '', help_text='basket containing this material components')
    basket_size_field = forms.IntegerField(label='Basket Size', initial = 5)
    basket_frequency_field = forms.CharField(label='Basket Freq/Support%', max_length=4, initial = '50%', required=True, help_text='Absolute Frequence or Relative Support%')
    def clean(self):
        if models.fdm.e2e.empty:
            self.add_error(None, "First retrieve your E2E data")
    def add_form_error(self, message):
        if not self._errors:
            self._errors = ErrorDict()
        if not NON_FIELD_ERRORS in self._errors:
            self._errors[NON_FIELD_ERRORS] = self.error_class()
        self._errors[NON_FIELD_ERRORS].append(message)

class genesis_form(forms.Form):
    input_file_field = forms.CharField(label='Input Filename', max_length=30, required = False, initial = '')
    data_list_file_field = forms.CharField(label='Data List Filename', max_length=30, required = False, initial = '')
    be_field = forms.FloatField(label='BE Value', required = False, help_text='Min Nr Form Count')
    project_field = forms.CharField(label='Project', max_length=30, required = False, initial = '875303WRA')
    nrexp_field = forms.IntegerField(label='Number of Experiment to Generate', initial = 10)
    def add_form_error(self, message):
        if not self._errors:
            self._errors = ErrorDict()
        if not NON_FIELD_ERRORS in self._errors:
            self._errors[NON_FIELD_ERRORS] = self.error_class()
        self._errors[NON_FIELD_ERRORS].append(message)

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
        
    def save(self, commit=True):
        user = super(RegistrationForm, self).save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            
        return user

