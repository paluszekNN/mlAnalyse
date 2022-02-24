import json

from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from .models import Data
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, SGDRegressor, Lars, Lasso
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
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

        df.dropna(inplace=True, axis=1)

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
        df.dropna(inplace=True, axis=1)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                df.drop(col, axis=1, inplace=True)
        lr = LinearRegression
        el1 = ElasticNet
        el05 = ElasticNet
        el025 = ElasticNet
        el01 = ElasticNet
        lasso = Lasso
        ridge = Ridge
        sgd = SGDRegressor
        lars = Lars
        list_of_elasticnet_alpha = [1, 0.5, 0.25, 0.1]

        models = [lr, el1, el05, el025, el01, lasso, ridge, sgd, lars]
        data_X = df.drop(df.columns[-1], axis=1)
        data_y = df[df.columns[-1]]
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        kf.get_n_splits(data_X)
        models_mean_score = []
        coefs = []
        biases = []
        model_names = []
        for model in models:
            scores = []
            if model == ElasticNet:
                alpha = list_of_elasticnet_alpha.pop()
            for train, test in kf.split(data_X):
                if model == ElasticNet:
                    model_to_fit = model(alpha)
                else:
                    model_to_fit = model()
                model_to_fit.fit(data_X.iloc[train], data_y.iloc[train])
                scores.append(r2_score(model_to_fit.predict(data_X.iloc[test]), data_y.iloc[test]))
            models_mean_score.append(np.mean(scores))
            if model == ElasticNet:
                model_to_fit = model(alpha)
            else:
                model_to_fit = model()
            model_to_fit.fit(data_X, data_y)
            biases.append(float(model_to_fit.intercept_))
            coefs.append(np.round(model_to_fit.coef_, decimals=4))
            model_names.append(str(model_to_fit))
        p = coefs[np.argmax(models_mean_score)].argsort()[::-1]
        for i in range(coefs.__len__()):
            coefs[i] = coefs[i][p]
        data = zip(model_names, np.round(models_mean_score, decimals=4), np.round(biases, decimals=4), coefs)

        context['data_0'] = data
        context['features'] = np.array(df.columns[:-1])[p]
        return context


class TSMView(generic.ListView):
    template_name = 'tsm.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True, axis=1)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                df.drop(col, axis=1, inplace=True)
        ex = ExtraTreeRegressor
        dtr = DecisionTreeRegressor

        models = [ex, dtr]
        data_X = df.drop(df.columns[-1], axis=1)
        data_y = df[df.columns[-1]]
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        kf.get_n_splits(data_X)
        models_mean_score = []
        coefs = []
        model_names = []
        for model in models:
            scores = []
            for train, test in kf.split(data_X):
                model_to_fit = model()
                model_to_fit.fit(data_X.iloc[train], data_y.iloc[train])
                scores.append(r2_score(model_to_fit.predict(data_X.iloc[test]), data_y.iloc[test]))
            models_mean_score.append(np.mean(scores))
            model_to_fit = model()
            model_to_fit.fit(data_X, data_y)
            coefs.append(np.round(model_to_fit.feature_importances_, decimals=4))
            model_names.append(str(model_to_fit))
        p = coefs[np.argmax(models_mean_score)].argsort()[::-1]
        for i in range(coefs.__len__()):
            coefs[i] = coefs[i][p]
        data = zip(model_names, np.round(models_mean_score, decimals=4), coefs)

        context['data_0'] = data
        context['features'] = np.array(df.columns[:-1])[p]
        return context
