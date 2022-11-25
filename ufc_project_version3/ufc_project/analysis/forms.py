from django import forms

class featuresform(forms.Form):
    CHOICES = [('bag', 'Bag of Words'), ('part', 'Parts of Speech Tagging'), ('tf', 'TF-IDF'),
               ('dp', 'Discrete Positive'), ('dn', 'Discrete Negative'), ('pol', 'Polarity'), ('sent', 'Sentiments'),
               ('all', 'All')
               ]

    feature = forms.CharField(label='Feature Type', widget=forms.RadioSelect(choices=CHOICES))



class preprocessingform(forms.Form):
    CHOICES = [('special', 'Special Character Removal + Stopwords'), ('lem', 'Lemmitization'), ('stop', 'Stopwords Removal')
               ]
    prep = forms.CharField(label='Feature Type', widget=forms.RadioSelect(choices=CHOICES))

class ml_modelform(forms.Form):
    CHOICES = [('nb', 'Naive Bayes'), ('rf', 'Random Forest')
               ]
    classifier = forms.CharField(label='Feature Type', widget=forms.RadioSelect(choices=CHOICES))
    eval_metrics = forms.CharField(max_length=100)
    val_tech = forms.CharField(max_length=100)

class registrationform(forms.Form):
    email = forms.CharField(max_length=80)
    password= forms.CharField(max_length=80,widget=forms.PasswordInput)


