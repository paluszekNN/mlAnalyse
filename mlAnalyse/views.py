from django.shortcuts import render
from django.contrib import messages
import pandas as pd
from linear_regression.models import Data, Analyse


def data_upload(request):
    template = 'index.html'

    prompt = {
        'order': ''
    }

    if request.method == "GET":
        return render(request, template, prompt)

    csv_file = request.FILES['file']
    sep = request.POST["sep"]
    label = request.POST["label"]

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a csv file')

    try:
        data = pd.read_csv(csv_file, sep=sep)

        set_label_as_last_column(label, data)
        data.dropna(inplace=True, axis=0)

        Data.objects.all().delete()
        Analyse.objects.all().delete()
        new_data = Data(name=csv_file, data=data.to_json())
        new_data.save()
    except:
        messages.error(request, 'This file can\'t export as data')


    context = {}
    return render(request, template, context)


def set_label_as_last_column(label, data: pd.DataFrame):
    data_y = data[label]
    data.drop(label, axis=1, inplace=True)
    data[label] = data_y