from django.shortcuts import render

from joblib import load
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = ['soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)


def index(req):
    model = load('./chatgroup/static/chatgroup.model')
    label = ""
    group = ""

    if req.method == 'POST':
        print("POST ")
        group = str(req.POST['group'])
        print(group)

        pred = model.predict([group])
        label = train.target_names[pred[0]]
        
    return render(req, 'chatgroup/index.html' ,{
            'label':label,
    })
