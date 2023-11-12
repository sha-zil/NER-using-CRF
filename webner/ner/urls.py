from django.urls import path
from . import views

urlpatterns = [
    # by default, when the base url is opened, it executes the begin method in views.py which then displays index.html
    path('', views.begin, name='begin'),
    # when the url matches .../initpredict as requested by the index.html page, it executes the train method in views.py which then displays predict.html
    path('initpredict',views.train,name='train'),
    # when the url matches .../predict as requested by the result.html page, it executes the predict method in views.py which then displays predict.html
    path('predict', views.predict, name='predict'),
    # when the url matches .../result, as requested by the predict.html page, it executes the result method in views.py which then displays result.html
    path('result', views.result, name='result'),
]