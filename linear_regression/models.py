from django.db import models


class Data(models.Model):
    name = models.CharField(max_length=100)
    data = models.JSONField()


class Analyse(models.Model):
    name = models.CharField(max_length=50)
    alpha = models.FloatField()
    name2 = models.CharField(max_length=50)
    alpha2 = models.FloatField()
