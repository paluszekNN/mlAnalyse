from django.db import models


class Data(models.Model):
    name = models.CharField(max_length=100)
    data = models.JSONField()
