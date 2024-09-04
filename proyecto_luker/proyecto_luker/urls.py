"""
URL configuration for proyecto_luker project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from .views import *



urlpatterns = [
    path('predict_sens/', PredictSensView.as_view(), name='predict_sens'),
    path('predict_fq/', PredictFqView.as_view(), name='predict_fq'),
      path('fetch-sens-predictions/', FetchSensPredictionsView.as_view(), name='fetch_sens_predictions'),
    path('fetch-fq-predictions/', FetchFqPredictionsView.as_view(), name='fetch_fq_predictions'),
    #path('input-sensorial-data/', InputSensorialDataView.as_view(), name='input-sensorial-data'),
    #path('fetch-sensorial-data/', FetchAllSensorialDataView.as_view(), name='fetch-sensorial-data'),
    path('retrain/', RetrainModelAPIView.as_view(), name='retrain_model_api'),
    # Add other paths here
]

