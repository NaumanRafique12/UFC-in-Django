from django.db import models
from django import forms

# Create your models here.

class features(models.Model):
    CHOICES = [('bag', 'Bag of Words'), ('part', 'Parts of Speech Tagging'), ('tf', 'TF-IDF'),
               ('dp', 'Discrete Positive'), ('dn', 'Discrete Negative'), ('pol', 'Polarity'), ('sent', 'Sentiments'),
               ('all', 'All')
               ]
    feature = models.CharField(max_length=500)


class preprocessing(models.Model):
    prep = models.CharField(max_length=500)

class ml_model(models.Model):

    classifier = models.CharField(max_length=500)
    accuracy = models.CharField(max_length=500)
    fmeasure = models.CharField(max_length=500)
    precision = models.CharField(max_length=500)
    recall = models.CharField(max_length=500)
    val_tech = models.CharField(max_length=100)

class registration(models.Model):
    name = models.CharField(max_length=80)
    email = models.CharField(max_length=80)
    username= models.CharField(max_length=80)
    password= models.CharField(max_length=80)

    def __str__(self):
        return self.name

class login(models.Model):
    email = models.CharField(max_length=80)
    password= models.CharField(max_length=80)








