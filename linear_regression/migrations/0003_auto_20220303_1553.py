# Generated by Django 3.2.8 on 2022-03-03 14:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('linear_regression', '0002_analyse'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analyse',
            name='alpha',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='analyse',
            name='alpha2',
            field=models.FloatField(),
        ),
    ]
