import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views import generic
from .models import Data, Analyse, Comparison
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, SGDRegressor, Lars, Lasso
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np


class DataView(generic.ListView):
    template_name = 'data.html'

    def get_queryset(self):
        return None

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True, axis=1)
        json_records = df.reset_index(drop=True).to_json(orient='records')
        data = json.loads(json_records)

        context['features'] = df.columns
        context['data'] = data
        return context


class ChartView(generic.ListView):
    template_name = 'comparison_of_columns.html'

    def get_queryset(self):
        return None

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        df.dropna(inplace=True, axis=1)
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                df.drop(col, axis=1, inplace=True)
        context['columns'] = df.columns
        df.sort_values(df.columns[-1], inplace=True)
        dfs = {}
        min = df.iloc[0][df.columns[-1]]
        max = df.iloc[-1][df.columns[-1]]
        if Comparison.objects.all():
            column = Comparison.objects.all()[0].column
            column2 = Comparison.objects.all()[0].column2
        else:
            column = df.columns[0]
            column2 = df.columns[1]
        print(column)
        for i in range(6):
            if i<5:
                dfs[str(i)] = df.loc[(df[df.columns[-1]]>= round(min+(max-min)*i/6,2)) & (df[df.columns[-1]]< round(min+(max-min)*(i+1)/6,2))][[column, column2]].reset_index(drop=True).to_json(orient='records')
                context['label'+str(i)] = str(df.columns[-1]) + ' (' +str(round(min+(max-min)*i/6,2))+ '-' + str(round(min+(max-min)*(i+1)/6,2)) + ')'
            else:
                dfs[str(i)] = df.loc[df[df.columns[-1]] >= round(min + (max - min) * i / 6, 2)][[column, column2]].reset_index(drop=True).to_json(orient='records')
                context['label'+str(i)] = str(df.columns[-1]) + ' (' +str(round(min+(max-min)*i/6,2))+ '-' + str(max) + ')'
            dfs[str(i)] = json.loads(dfs[str(i)])

            context['data'+str(i)] = dfs[str(i)]

        context['feature1'] = column
        context['feature2'] = column2

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

        scaler = StandardScaler()
        df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
        self.data_X = df.drop(df.columns[-1], axis=1)
        self.data_y = df[df.columns[-1]]
        self.kf = KFold(n_splits=5, random_state=None, shuffle=True)
        self.kf.get_n_splits(self.data_X)

    @property
    def type_of_modeling(self) -> TypeOfModeling:
        return self._type_of_modeling

    @type_of_modeling.setter
    def type_of_modeling(self, type_of_modeling: TypeOfModeling) -> None:
        self._type_of_modeling = type_of_modeling

    def get_attributes(self):
        attributes, models_mean_score, model_names = self.collect_attributes()

        assert isinstance(attributes[0][0], np.ndarray)
        sorted_indexes_by_score = attributes[0][np.argmax(models_mean_score)].argsort()[::-1]

        self.sort_features_importances_values(attributes, sorted_indexes_by_score)

        return (model_names, *attributes, models_mean_score), sorted_indexes_by_score

    def collect_attributes(self):
        model_names = []
        models_mean_score = []

        list_of_elastic_net_alpha, models = self._type_of_modeling.get_models()
        features_importances = []
        intercepts = []

        for model in models:
            el_alpha = None
            if model == ElasticNet:
                el_alpha = list_of_elastic_net_alpha.pop()
            scores = self.get_scores_from_kfold_split(model, el_alpha)
            models_mean_score.append(np.round(np.mean(scores), decimals=4))
            if model == ElasticNet:
                model_to_fit = model(el_alpha)
            else:
                model_to_fit = model()
            model_to_fit.fit(self.data_X, self.data_y)
            attributes = self._type_of_modeling.get_attributes(model_to_fit)
            features_importances.append(np.round(attributes[0], decimals=4))
            if self._type_of_modeling.__class__ == LinearModels:
                intercepts.append(np.round(attributes[1], decimals=4))
            model_names.append(str(model_to_fit))
        if intercepts:
            attributes = (features_importances, intercepts)
        else:
            attributes = (features_importances,)
        return attributes, models_mean_score, model_names

    def get_scores_from_kfold_split(self, model, el_alpha):
        scores = []
        for train, test in self.kf.split(self.data_X):
            if model == ElasticNet:
                model_to_fit = model(el_alpha)
            else:
                model_to_fit = model()
            model_to_fit.fit(self.data_X.iloc[train], self.data_y.iloc[train])
            scores.append(r2_score(model_to_fit.predict(self.data_X.iloc[test]), self.data_y.iloc[test]))
        return scores

    def sort_features_importances_values(self, attributes, sorted_indexes_by_score):
        for i in range(attributes[0].__len__()):
            attributes[0][i] = attributes[0][i][sorted_indexes_by_score]


class LRView(generic.ListView):
    template_name = 'lr.html'

    def get_queryset(self):
        return None

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)

        ml = MLModels(LinearModels())
        attributes, sorted_indexes_by_score = ml.get_attributes()

        for att in attributes:
            assert isinstance(att, list)

        context['attributes'] = zip(*attributes)
        context['features'] = np.array(ml.data_X.columns)[sorted_indexes_by_score]
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
        context['features'] = np.array(ml.data_X.columns)[sorted_indexes_by_score]
        return context


class AnalyseView(generic.ListView):
    template_name = 'summarization.html'

    def get_queryset(self):
        return None

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        ml = MLModels(LinearModels())
        list_of_elasticnet_alpha, linear_models = ml.type_of_modeling.get_models()
        ml.type_of_modeling = TreeModels()
        _, tree_models = ml.type_of_modeling.get_models()
        models = list(set(linear_models)) + tree_models
        context['models'] = sorted(models, key=lambda x: str(x))
        context['list_of_elasticnet_alpha'] = list_of_elasticnet_alpha

        if Analyse.objects.all():
            analyse = Analyse.objects.all()[0]
            models_to_sum = {}
            for model in models:
                if str(model()) == analyse.name:
                    models_to_sum['first_model'] = {'object': model()}
                    models_to_sum['first_model']['object'].fit(ml.data_X, ml.data_y)

                elif str(model()) == analyse.name2:
                    models_to_sum['second_model'] = {'object': model()}
                    models_to_sum['second_model']['object'].fit(ml.data_X, ml.data_y)

            for model in models_to_sum.keys():
                if 'tree' in str(models_to_sum[model]['object']).lower():
                    feature_scores = models_to_sum[model]['object'].feature_importances_
                else:
                    feature_scores = models_to_sum[model]['object'].coef_
                sorted_indexes_by_score = np.abs(feature_scores).argsort()[::-1]
                sorted_features = ml.data_X.columns[sorted_indexes_by_score]
                context[model + '_best_features'] = ', '.join(list(sorted_features)[:(len(ml.data_X.columns)) // 5 + 1])
                worst_features = list(sorted_features)[-(len(ml.data_X.columns)) // 5:]
                worst_features.reverse()
                context[model + '_worst_features'] = ', '.join(worst_features)
                context[model] = str(models_to_sum[model]['object']).split('(')[0]
            context['features'] = ml.data_X.columns
        return context


def save_model_to_compare(request):
    template = 'summarization.html'

    model = request.POST["model"]
    alpha = request.POST["alpha"]
    model2 = request.POST["model2"]
    alpha2 = request.POST["alpha2"]

    Analyse.objects.all().delete()
    new_analyse = Analyse(name=model, alpha=alpha, name2=model2, alpha2=alpha2)
    new_analyse.save()

    return redirect('summarization')


def save_columns_to_compare(request):
    template = 'comparison_of_columns.html'

    column = request.POST["column"]
    column2 = request.POST["column2"]

    Comparison.objects.all().delete()
    new_analyse = Comparison(column=column, column2=column2)
    new_analyse.save()

    return redirect('comparison')