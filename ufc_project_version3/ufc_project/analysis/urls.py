from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('index',views.index,name='index'),
    path('index2',views.index2,name='index'),
    path('delrec',views.delrecord,name='delete_record'),
    path('about',views.about,name='about'),
    path('result',views.myresult,name='resulttt'),
    path('dataset2',views.dataset_2,name='data_set2'),
    path('history',views.history,name='history'),
    path('registration',views.registration,name='registration'),
    path('register2',views.registration2,name='registration2'),
    path('error',views.error,name='error'),
    path('login',views.login,name='login'),
    path('login_check',views.login_check,name='logincheck'),
    path('home',views.homepage,name='homep'),
    path('dataset',views.dataset,name='dataset'),
    path('contact_us',views.contact_us,name='contact_us'),
    path('classifier',views.classifier,name='classifier'),
    path('unseen_review',views.unseen_review,name='unseen_review'),
    path('result',views.results,name='results'),

    path('sendemail',views.mail,name='mailss'),

]