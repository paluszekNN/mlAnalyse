import json

from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from .models import Data
import pandas as pd


class IndexView(generic.ListView):
    template_name = 'ml.html'
    context_object_name = 'data'

    def get_queryset(self):
        return Data.objects.all()

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        df = pd.DataFrame(json.loads(Data.objects.all()[0].data))
        print(df.head(1))
        df.dropna(inplace=True)

        context['features'] = df.columns
        json_records = df.reset_index(drop=True).to_json(orient='records')
        data = json.loads(json_records)
        context['data'] = data
        return context



