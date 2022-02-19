from django.urls import path

from . import views

urlpatterns = [
    path('', views.DataView.as_view(), name='data'),
    path('lr', views.LRView.as_view(), name='lr'),
]