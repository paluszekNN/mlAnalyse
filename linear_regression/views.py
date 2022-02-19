import json

from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from .models import Data
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, SGDRegressor, Lars, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np


class DataView(generic.ListView):
    template_name = 'data.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True)

        context['features'] = df.columns
        json_records = df.reset_index(drop=True).to_json(orient='records')
        data = json.loads(json_records)
        context['data'] = data
        return context


class LRView(generic.ListView):
    template_name = 'lr.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True)
        for col in df.columns:
            try:
                df[col].astype(float)
            except:
                df.drop(col, axis=1, inplace=True)
        lr = LinearRegression()
        el1 = ElasticNet(1)
        el05 = ElasticNet(0.5)
        el025 = ElasticNet(0.25)
        el01 = ElasticNet(0.1)
        lasso = Lasso()
        ridge = Ridge()
        sgd = SGDRegressor()
        lars = Lars()

        models = [lr, el1, el05, el025, el01, lasso, ridge, sgd, lars]
        data_X = df.drop(df.columns[-1], axis=1)
        data_y = df[df.columns[-1]]
        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        kf.get_n_splits(data_X)
        models_mean_score = []
        coefs = []
        biases = []
        model_names = []
        for model in models:
            scores = []
            for train, test in kf.split(data_X):
                model.fit(data_X.iloc[train], data_y.iloc[train])
                scores.append(r2_score(model.predict(data_X.iloc[test]), data_y.iloc[test]))
            models_mean_score.append(np.mean(scores))
            coefs.append(np.round(model.coef_, decimals=4))
            biases.append(float(model.intercept_))
            model_names.append(str(model))
        p = coefs[0].argsort()[::-1]
        for i in range(coefs.__len__()):
            coefs[i] = coefs[i][p]
        data = zip(model_names, np.round(models_mean_score, decimals=4), np.round(biases, decimals=4), coefs)

        context['data_0'] = data
        context['features'] = np.array(df.columns[:-1])[p]
        return context
