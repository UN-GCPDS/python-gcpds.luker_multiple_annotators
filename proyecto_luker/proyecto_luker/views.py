from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import numpy as np
import tensorflow as tf # Import the specific class
from .multiple_annotators_models import MA_GCCE  # Import the specific class or function
from .parameters import SENS_VARS_CHOC  # Ensure this import is from the correct relative path
import tensorflow_probability as tfp
import os
from recetario_luker.models import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
import keras


class GCCE_MA_lossPredict:
    def __init__(self, q=0.1, R=5, K=10):
        self.q = q
        self.R = R
        self.K = K
    @keras.saving.register_keras_serializable()
    def custom_loss(self, y_true, y_pred):
        print('Este es el shape')
        print(y_true.shape,y_pred.shape)
        pred = y_pred[:, self.R:]
        pred = tf.clip_by_value(pred, clip_value_min=1e-9, clip_value_max=1)
        ann_ = y_pred[:, :self.R]
        
        Y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=self.K, axis=1)
        Y_hat = tf.repeat(tf.expand_dims(pred, -1), self.R, axis=-1)

        p_gcce = Y_true * (1 - Y_hat**self.q) / self.q
        temp1 = ann_ * tf.math.reduce_sum(p_gcce, axis=1)
        temp2 = (1 - ann_) * (1 - (1/self.K)**self.q) / self.q * tf.reduce_sum(Y_true, axis=1)
        
        return tf.math.reduce_sum(temp1 + temp2)
    def get_config(self):
        """Return the config of the class for serialization."""
        return {
            'q': self.q,
            'R': self.R,
            'K': self.K
        }

    @classmethod
    def from_config(cls, config):
        """Instantiate the class from the config."""
        return cls(**config)


class RetrainModelAPIView(APIView):
    def post(self, request):
        # Get the parameters from the POST request body
        q = float(request.data.get('q', 0.1))  # Default value 0.1 if not provided
        R = int(request.data.get('R', 5))     # Default value 5 if not provided
        K = int(request.data.get('K', 10))    # Default value 10 if not provided

        # Load data from the FqPrediction model
        data = FqPrediction.objects.all()

        # Prepare the data for training
        X = np.array([np.array([item.acidez, item.amargor, item.aroma, item.astringencia, 
                       item.dulce, item.dureza, item.impresion, item.fusion]) for item in data])

        # Extract the results as y (assuming it's stored as a list in JSON format)
        y = np.array([np.array(item.result) for item in data])
        print(y.shape)
        y = np.squeeze(y, axis=1)  # This will remove the second dimension (the singleton dimension)
    
        # Initialize custom loss with dynamic parameters from the POST request
        loss_instance = GCCE_MA_lossPredict(q, R, K)

        # Initialize your model (assuming you want to retrain MA_GCCE)
        model = tf.keras.models.load_model(
            os.path.join(base_path, 'models/gcce/model_s2fq.keras'),
           
        )

        # Compile the model with the desired parameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025)
        model.compile(optimizer=optimizer, loss=loss_instance.custom_loss, metrics=['accuracy'])

        # Retrain the model
        model.fit(X, y)

        # Optionally save the retrained model
        model.save(os.path.join(base_path,'models/gcce/model_s2fq.keras'))

        return Response({"message": "Model retrained successfully!"}, status=status.HTTP_200_OK)
            
# View to fetch all SensPrediction records
class FetchSensPredictionsView(APIView):
    def get(self, request):
        predictions = SensPrediction.objects.all()
        data = [
            {
                "id": prediction.id,
                "humedad": prediction.humedad,
                "grasa": prediction.grasa,
                "granulometria": prediction.granulometria,
                "viscosidad": prediction.viscosidad,
                "fluidez": prediction.fluidez,
                "result": prediction.result,
                'anotador': prediction.annotator_code,
                "created_at": prediction.created_at
            }
            for prediction in predictions
        ]
        return Response(data, status=status.HTTP_200_OK)

# View to fetch all FqPrediction records
class FetchFqPredictionsView(APIView):
    def get(self, request):
        predictions = FqPrediction.objects.all()
        data = [
            {
                "id": prediction.id,
                "acidez": prediction.acidez,
                "amargor": prediction.amargor,
                "aroma": prediction.aroma,
                "astringencia": prediction.astringencia,
                "dulce": prediction.dulce,
                "dureza": prediction.dureza,
                "impresion": prediction.impresion,
                "fusion": prediction.fusion,
                "result": prediction.result,
                'anotador': prediction.annotator_code,
                "created_at": prediction.created_at.isoformat() 
            }
            for prediction in predictions
        ]
        return Response(data, status=status.HTTP_200_OK)

class MultiOutputModel(tf.keras.Model):
    def __init__(self, models):
        super(MultiOutputModel, self).__init__()
        self.models = models 

    def predict(self, inputs, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        # Override the predict method
        return np.vstack([np.argmax(model.predict(inputs)[:,6:], axis=1) for model in self.models]).T
    



# Define the custom loss function using the imported class/function
def GCCE_MA_loss(y_true, y_pred):
    return MA_GCCE.GCCE_MA_loss(y_true, y_pred)
base_path = os.path.dirname(__file__)  # Get the directory where the current file is located
# Load the sensory models
def load_model_sens():
    models_sens = []
    base_path = os.path.dirname(__file__)  # Get the directory where the current file is located

    for var in SENS_VARS_CHOC:
        model_path = os.path.join(base_path, f"models/gcce/gcce_ma_{var}.keras")
        models_sens.append(tf.keras.models.load_model(
            model_path,
            custom_objects={'method': GCCE_MA_loss}
        ))

    return MultiOutputModel(models_sens)

# Load the models when the script is run




@method_decorator(csrf_exempt, name='dispatch')
class PredictSensView(APIView):
    def get(self, request):
        # Extract inputs
        model_to_sens = load_model_sens()
        humedad = float(request.GET.get('humedad'))
        grasa = float(request.GET.get('grasa'))
        granulometria = float(request.GET.get('granulometria'))
        viscosidad = float(request.GET.get('viscosidad'))
        fluidez = float(request.GET.get('fluidez'))
        anotador = str(request.GET.get('anotador'))
        
        # Prepare input data for prediction
        input_data = np.array([[humedad, grasa, granulometria, viscosidad, fluidez]])
        predictions = model_to_sens.predict(input_data)

        # Save inputs and prediction result to the database
        sens_prediction = SensPrediction.objects.create(
            humedad=humedad,
            grasa=grasa,
            granulometria=granulometria,
            viscosidad=viscosidad,
            fluidez=fluidez,
            annotator_code=anotador,
            result=predictions.tolist()
        )

        # Return the prediction result
        return JsonResponse({"sens": predictions.tolist()}, status=status.HTTP_200_OK)

@method_decorator(csrf_exempt, name='dispatch')
class PredictFqView(APIView):
    def get(self, request):
        # Extract inputs
        model_to_fq = tf.keras.models.load_model(os.path.join(base_path,'models/gcce/model_s2fq.keras'))
        acidez = float(request.GET.get('acidez'))
        amargor = float(request.GET.get('amargor'))
        aroma = float(request.GET.get('aroma'))
        astringencia = float(request.GET.get('astringencia'))
        dulce = float(request.GET.get('dulce'))
        dureza = float(request.GET.get('dureza'))
        impresion = float(request.GET.get('impresion'))
        fusion = float(request.GET.get('fusion'))
        anotador = str(request.GET.get('anotador'))

        # Prepare input data for prediction
        input_data = np.array([[acidez, amargor, aroma, astringencia, dulce, dureza, impresion, fusion]])
        predictions = model_to_fq.predict(input_data)

        # Save inputs and prediction result to the database
        fq_prediction = FqPrediction.objects.create(
            acidez=acidez,
            amargor=amargor,
            aroma=aroma,
            astringencia=astringencia,
            dulce=dulce,
            dureza=dureza,
            impresion=impresion,
            fusion=fusion,
            annotator_code=anotador,
            result=predictions.tolist()
        )

        # Return the prediction result
        return JsonResponse({"fq": predictions.tolist()}, status=status.HTTP_200_OK)
    


# View to input data
class InputSensorialDataView(APIView):
    def post(self, request):
        # Extract data from the request
        humedad = request.data.get('humedad')
        grasa = request.data.get('grasa')
        granulometria = request.data.get('granulometria')
        viscosidad = request.data.get('viscosidad')
        fluidez = request.data.get('fluidez')
        annotator_code = request.data.get('annotator_code')

        # Validate the required fields
        if not all([humedad, grasa, granulometria, viscosidad, fluidez, annotator_code]):
            return Response({"error": "All fields are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Create a new SensorialData object
        sensorial_data = SensorialData.objects.create(
            humedad=humedad,
            grasa=grasa,
            granulometria=granulometria,
            viscosidad=viscosidad,
            fluidez=fluidez,
            annotator_code=annotator_code
        )

        return Response({"message": "Sensorial data created successfully.", "id": sensorial_data.id}, status=status.HTTP_201_CREATED)
    

# View to fetch all data
class FetchAllSensorialDataView(APIView):
    def get(self, request):
        # Fetch all records from SensorialData model
        sensor_data = SensorialData.objects.all()
        
        # Serialize the data into a list of dictionaries
        data = [
            {
                "id": item.id,
                "humedad": item.humedad,
                "grasa": item.grasa,
                "granulometria": item.granulometria,
                "viscosidad": item.viscosidad,
                "fluidez": item.fluidez,
                "created_at": item.created_at.isoformat(),
                "annotator_code": item.annotator_code
            }
            for item in sensor_data
        ]

        return JsonResponse(data, safe=False, status=status.HTTP_200_OK)