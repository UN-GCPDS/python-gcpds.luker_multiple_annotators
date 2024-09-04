
import numpy as np
from django.db import models




class SensPrediction(models.Model):
    humedad = models.FloatField()
    grasa = models.FloatField()
    granulometria = models.FloatField()
    viscosidad = models.FloatField()
    fluidez = models.FloatField()
    result = models.JSONField()  # Store the prediction result as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    annotator_code = models.TextField(null=False)


class FqPrediction(models.Model):
    acidez = models.FloatField()
    amargor = models.FloatField()
    aroma = models.FloatField()
    astringencia = models.FloatField()
    dulce = models.FloatField()
    dureza = models.FloatField()
    impresion = models.FloatField()
    fusion = models.FloatField()
    result = models.JSONField()  # Store the prediction result as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    annotator_code = models.TextField(null=False)

class SensorialData(models.Model):
    humedad = models.FloatField()
    grasa = models.FloatField()
    granulometria = models.FloatField()
    viscosidad = models.FloatField()
    fluidez = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    annotator_code = models.TextField(null=False)
