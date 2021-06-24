# Team AI

## Installation
1. `conda create -n django_services python=3.6` (3.6 due to some AI library support)
2. `conda activate django_services`
3. `pip install -r requirements.txt`
4. `python manage.py runserver 0.0.0.0:8000`

## Applications routes

***

**Web services**

*/web_services*

This endpoint is for end-user applications

- *GET* /web_services/detect_facial_emotions

This application takes a snapshot with webcam of a face and predicts the emotion present in the face among the following tags: 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise' or 'Neutral'. This is a refactoring of [this project](https://colab.research.google.com/drive/1V7XMG9CB6zreYzURlE785ZECBob7NX4L#scrollTo=8Regb5LGlBv-)

- *POST* /web_services/detect_facial_emotions

You can send a PNG image in base64 format to this endpoint, as result it returns a json with the face emotion prediction

> JSON Request

| Name          | Type             | 
| ------------- |:----------------:| 
| image         | Base64 PNG image | 

> JSON Response

| Name          | Type             | 
| ------------- |:----------------:| 
| prediction    | String           | 

***

**Projects**

*/projects*

This endpoint is to show projects that cant be used directly by an user

- *GET* /projects/neat

This project is about Neural Networks and genetic algorithms. Is composed by two parts:
1. A set of agents that learns to play Flappy birds
> [Based on this video](https://www.youtube.com/watch?v=OGHA-elMrxI)
2. A set of agents that learns to collect life and try to avoid collision with an enemy, who is walking randomly
> [See the Demo of this project here](https://drive.google.com/file/d/1a_AYG1VFoml3hF0QJxaw9QAQu0b7nP4e/view?usp=sharing)

***

**Games**

*/games*

This endpoint is for video games made in Unity that actually use artificial intelligence

- *GET* /games/text101

This application is just a demo/example to be used as reference to setup a Unity game in this app endpoint

***

## Extra configuration

> .vscode
---
```
{
    "python.analysis.extraPaths": ["neat"]
}
```
