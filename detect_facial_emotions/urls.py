from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="detect_facial_emotions"),
    path("train", views.train, name="train_facial_emotions")
]

