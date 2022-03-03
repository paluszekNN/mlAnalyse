import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views import generic
from .models import Data, Analyse
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


class TypeOfModeling:
    def get_models(self):
        pass

    def get_attributes(self, model):
        pass


class LinearModels(TypeOfModeling):
    def get_models(self):
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
        return list_of_elasticnet_alpha, models

    def get_attributes(self, model):
        return model.coef_, model.intercept_


class TreeModels(TypeOfModeling):
    def get_attributes(self, model):
        return model.feature_importances_,

    def get_models(self):
        ex = ExtraTreeRegressor
        dtr = DecisionTreeRegressor

        models = [ex, dtr]
        return None, models


class MLModels:
    def __init__(self, type_of_modeling):
        self._type_of_modeling = type_of_modeling
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True, axis=1)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                df.drop(col, axis=1, inplace=True)
        self.data = df

    @property
    def type_of_modeling(self) -> TypeOfModeling:
        return self._type_of_modeling

    @type_of_modeling.setter
    def type_of_modeling(self, type_of_modeling: TypeOfModeling) -> None:
        self._type_of_modeling = type_of_modeling

    def get_attributes(self):
        data_X = self.data.drop(self.data.columns[-1], axis=1)
        data_y = self.data[self.data.columns[-1]]
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        kf.get_n_splits(data_X)

        attributes = {}
        model_names = []
        models_mean_score = []

        list_of_elastic_net_alpha, models = self._type_of_modeling.get_models()
        for model in models:
            scores = []
            el_alpha = None
            if model == ElasticNet:
                el_alpha = list_of_elastic_net_alpha.pop()
            for train, test in kf.split(data_X):
                if model == ElasticNet:
                    model_to_fit = model(el_alpha)
                else:
                    model_to_fit = model()
                model_to_fit.fit(data_X.iloc[train], data_y.iloc[train])
                scores.append(r2_score(model_to_fit.predict(data_X.iloc[test]), data_y.iloc[test]))
            models_mean_score.append(np.round(np.mean(scores), decimals=4))
            if model == ElasticNet:
                model_to_fit = model(el_alpha)
            else:
                model_to_fit = model()
            model_to_fit.fit(data_X, data_y)
            i = 0
            for att in self._type_of_modeling.get_attributes(model_to_fit):
                if str(i) not in attributes.keys():
                    attributes[str(i)] = []
                attributes[str(i)].append(np.round(att, decimals=4))
                i += 1
            model_names.append(str(model_to_fit))
        assert isinstance(attributes['0'][0], np.ndarray)
        sorted_indexes_by_score = attributes['0'][np.argmax(models_mean_score)].argsort()[::-1]

        for i in range(attributes['0'].__len__()):
            attributes['0'][i] = attributes['0'][i][sorted_indexes_by_score]

        return (model_names, *attributes.values(), models_mean_score), sorted_indexes_by_score


class LRView(generic.ListView):
    template_name = 'lr.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)

        ml = MLModels(LinearModels())
        attributes, sorted_indexes_by_score = ml.get_attributes()

        for att in attributes:
            assert isinstance(att, list)

        context['attributes'] = zip(*attributes)
        context['features'] = np.array(ml.data.columns[:-1])[sorted_indexes_by_score]
        return context


class TSMView(generic.ListView):
    template_name = 'tsm.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)

        ml = MLModels(TreeModels())
        attributes, sorted_indexes_by_score = ml.get_attributes()
        context['attributes'] = zip(*attributes)
        context['features'] = np.array(ml.data.columns[:-1])[sorted_indexes_by_score]
        return context


class AnalyseView(generic.ListView):
    template_name = 'summarization.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        ml = MLModels(LinearModels())
        list_of_elasticnet_alpha, linear_models = ml.type_of_modeling.get_models()
        ml.type_of_modeling = TreeModels()
        _, tree_models = ml.type_of_modeling.get_models()
        context['models'] = sorted(list(set(linear_models))+tree_models, key=lambda x:str(x))
        context['list_of_elasticnet_alpha'] = list_of_elasticnet_alpha

        # context['features'] = df.columns
        return context


def save_model_to_compare(request):
    template = 'summarization.html'

    model = request.POST["model"]
    alpha = request.POST["alpha"]
    model2 = request.POST["model2"]
    alpha2 = request.POST["alpha2"]
    print(alpha2)
    Analyse.objects.all().delete()
    new_analyse = Analyse(name=model, alpha=alpha, name2=model2, alpha2=alpha2)
    new_analyse.save()

    return redirect('summarization')