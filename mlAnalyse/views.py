from django.shortcuts import render
from django.contrib import messages
import csv, io
import pandas as pd


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
        print(data.head(2))
    except:
        messages.error(request, 'This file can\'t export as data')

    context = {}
    return render(request, template, context)