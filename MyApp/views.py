from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
from rest_framework.response import Response
from sklearn import tree
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import json
import numpy as np

@api_view(['POST'])
def formresult(request, format=None):
    x=request.data
    pH=x["pH"]
    Turbidity=x["Turbidity"]
    ElectricalConductivity=x["ElectricalConductivity"]
    Tds=x["Tds"]
    f = open("./MyApp/sample.txt","r")
    features=[]
    labels=[]
    co=f.readlines()
    try:
        for i in range(0, len(co)):
            co1=co[i].split(',')
            tf=[]
            for j in range (0, len(co1)-1):
                tf.append(float(co1[j]))
            features.append(tf)
            ts=co1[len(co1)-1]
            labels.append(ts[:-1])

    except EOFError:
        pass

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    result=clf.predict([[pH,Turbidity,ElectricalConductivity,Tds]])
    return Response(result)

@api_view(['GET'])
def comparison(request):
    # load dataset
    f = open("./MyApp/sample.txt","r")
    features=[]
    labels=[]
    x=[]
    y=[]
    co=f.readlines()
    try:
        for i in range(0, len(co)):
            co1=co[i].split(',')
            tf=[]
            for j in range (0, len(co1)-1):
                tf.append(float(co1[j]))
            features.append(tf)
            ts=co1[len(co1)-1]
            labels.append(ts[:-1])

    except EOFError:
        pass
    
    X=features
    Y=labels

    for i in range(len(Y)):
        if Y[i]=='pure' :Y[i]=float(0)
        elif Y[i]=='impure':Y[i]=float(1)
        elif Y[i]=='unusable' or Y[i]=='unusabl':Y[i]=float(2)
    Y=np.array(Y)

    # prepare models
    models = []
    models.append(('Decison Tree Classifier', DecisionTreeClassifier()))
    models.append(('Multilayer Perceptron Classifier', MLPClassifier()))
    models.append(('Random Forest Classifier', RandomForestClassifier()))

    comparison_result = []


    for name, model in models:
        
        kfold = model_selection.KFold(n_splits=10)
        y_pred = model_selection.cross_val_predict(model, X, Y, cv=kfold)
        
        #accuracy
        acc_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy') #acc
        comparison_result.append(acc_results.mean())
        comparison_result.append(acc_results.std())
        
        #f1_Score
        f1score =  metrics.f1_score(Y, y_pred, average='micro')
        comparison_result.append(f1score)
        
        #Mean Absolute Eror
        y_pred1=y_pred.astype(np.float64)
        mae =  metrics.mean_absolute_error(Y, y_pred1)
        comparison_result.append(mae)
        
        #Lograthmic Loss
        y_pred2 = model_selection.cross_val_predict(model, X, Y, cv=kfold,method='predict_proba')
        logscore =  metrics.log_loss(Y, y_pred2)
        comparison_result.append(logscore)
        #Mean Squared Error
        mse =   metrics.mean_squared_error(Y, y_pred)
        comparison_result.append(mse)
        
        #Area under curve
        fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred, pos_label=2)
        aucs = metrics.auc(fpr, tpr)
        comparison_result.append(aucs)

    return Response(comparison_result)
    



    
  



