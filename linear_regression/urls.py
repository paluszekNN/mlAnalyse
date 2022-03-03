from django.urls import path

from . import views

urlpatterns = [
    path('', views.DataView.as_view(), name='data'),
    path('lr', views.LRView.as_view(), name='lr'),
    path('tsm', views.TSMView.as_view(), name='tsm'),
    path('summarization', views.AnalyseView.as_view(), name='summarization'),
    path('save_analyse', views.save_model_to_compare, name='save_analyse'),
]