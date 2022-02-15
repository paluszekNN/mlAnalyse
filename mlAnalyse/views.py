from django.shortcuts import render
from django.contrib import messages
import pandas as pd
from linear_regression.models import Data


def data_upload(request):
    template = 'index.html'

    prompt = {
        'order': ''
    }

    if request.method == "GET":
        return render(request, template, prompt)

    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a csv file')

    try:
        data = pd.read_csv(csv_file)
    except:
        messages.error(request, 'This file can\'t export as data')

    Data.objects.all().delete()
    new_data = Data(name=csv_file, data=data.to_json())
    new_data.save()

    context = {}
    return render(request, template, context)