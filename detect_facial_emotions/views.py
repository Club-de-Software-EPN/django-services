from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import os
from .modules.module_detect_facial_emotions.train import train as facial_train
from .modules.module_detect_facial_emotions.models.EmotionClassifier import EmotionClassifier
from pathlib import Path
import io
import base64
from PIL import Image
import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent

CURRENT_DIR = f"{BASE_DIR}/detect_facial_emotions"
print(CURRENT_DIR)

model = keras.models.load_model(os.path.join('detect_facial_emotions', 'data', 'saved_models', 'facial_1.h5'))
# model = EmotionClassifier.get_full_model_nn()
# model.load_weights(os.path.join('detect_facial_emotions', 'data', 'saved_models', 'fer.h5'))



def index(request):
    #return HttpResponseRedirect(reverse("myurlname", args=["a"]))

    if request.method == "GET":
        return render(request, "detect_facial_emotions/index.html")
    elif request.method == "POST":
        emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

        

        image_base_64 = request.POST["image"].replace("data:image/png;base64,", "")

        base64_decoded = base64.b64decode(image_base_64)
        image_gray = Image.open(io.BytesIO(base64_decoded)).convert('L').resize((48,48))

        np_image_gray = np.array(image_gray)
        sample = np_image_gray.reshape(1, 48, 48, 1)
        # real = emo[np.argmax(y_train[0])]
        # print("real", real)


        prediction = emo[np.argmax(model.predict(sample))]

        return JsonResponse({"prediction": prediction})

def train():
    # facial_train(data_filepath, save_model_path, num_epochs=1, plot_history = False, verbose = 10):
    return HttpResponse("Not supported train throught UI")