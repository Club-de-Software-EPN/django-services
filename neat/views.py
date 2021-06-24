from django.shortcuts import render
from pathlib import Path
# Create your views here.

BASE_DIR = Path(__file__).resolve().parent.parent

CURRENT_DIR = f"{BASE_DIR}/detect_facial_emotions"
print(CURRENT_DIR)

def index(request):
    #return HttpResponseRedirect(reverse("myurlname", args=["a"]))

    if request.method == "GET":
        return render(request, "neat/index.html")