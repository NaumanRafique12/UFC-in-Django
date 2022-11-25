

from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .forms import featuresform, preprocessingform, ml_modelform, registrationform
from analysis.models import registration, features, ml_model, preprocessing


def select_fight_row(df, name, i):
    df_temp = df[(df['R_fighter'] == name) | (df['B_fighter'] == name)]  # filter df on fighter's name
    df_temp.reset_index(drop=True, inplace=True) #  as we created a new temporary dataframe, we have to reset indexes
    idx = max(df_temp.index)  #  get the index of the oldest fight
    if i > idx:  #  if we are looking for a fight that didn't exist, we return nothing
        return
    arr = df_temp.iloc[i,:].values
    return arr

def list_fighters(df, limit_date):
    df_temp = df[df['date'] > limit_date]
    set_R = set(df_temp['R_fighter'])
    set_B = set(df_temp['B_fighter'])
    fighters = list(set_R.union(set_B))
    return fighters

def build_df(df, fighters, i):
    arr = [select_fight_row(df, fighters[f], i) for f in range(len(fighters)) if select_fight_row(df, fighters[f], i) is not None]
    cols = [col for col in df]
    df_fights = pd.DataFrame(data=arr, columns=cols)
    df_fights.drop_duplicates(inplace=True)
    df_fights['title_bout'] = df_fights['title_bout'].replace({True: 1, False: 0})
    df_fights.drop(['R_fighter', 'B_fighter', 'date'], axis=1, inplace=True)
    return df_fights


def predict(df, pipeline, blue_fighter, red_fighter, weightclass, rounds, title_bout=False):
    # We build two dataframes, one for each figther
    f1 = df[(df['R_fighter'] == blue_fighter) | (df['B_fighter'] == blue_fighter)].copy()
    f1.reset_index(drop=True, inplace=True)
    f1 = f1[:1]
    f2 = df[(df['R_fighter'] == red_fighter) | (df['B_fighter'] == red_fighter)].copy()
    f2.reset_index(drop=True, inplace=True)
    f2 = f2[:1]

    # if the fighter was red/blue corner on his last fight, we filter columns to only keep his statistics (and not the other fighter)
    # then we rename columns according to the color of  the corner in the parameters using re.sub()
    if (f1.loc[0, ['R_fighter']].values[0]) == blue_fighter:
        result1 = f1.filter(regex='^R', axis=1).copy()  # here we keep the red corner stats
        result1.rename(columns=lambda x: re.sub('^R', 'B', x),
                       inplace=True)  # we rename it with "B_" prefix because he's in the blue_corner
    else:
        result1 = f1.filter(regex='^B', axis=1).copy()
    if (f2.loc[0, ['R_fighter']].values[0]) == red_fighter:
        result2 = f2.filter(regex='^R', axis=1).copy()
    else:
        result2 = f2.filter(regex='^B', axis=1).copy()
        result2.rename(columns=lambda x: re.sub('^B', 'R', x), inplace=True)

    fight = pd.concat([result1, result2], axis=1)  # we concatenate the red and blue fighter dataframes (in columns)
    fight.drop(['R_fighter', 'B_fighter'], axis=1, inplace=True)  # we remove fighter names
    fight.insert(0, 'title_bout',
                 title_bout)  # we add tittle_bout, weight class and number of rounds data to the dataframe
    fight.insert(1, 'weight_class', weightclass)
    fight.insert(2, 'no_of_rounds', rounds)
    fight['title_bout'] = fight['title_bout'].replace({True: 1, False: 0})

    pred = pipeline.predict(fight)
    proba = pipeline.predict_proba(fight)
    if (pred == 1.0):
        print("The predicted winner is", red_fighter, 'with a probability of', round(proba[0][1] * 100, 2), "%")
    else:
        print("The predicted winner is", blue_fighter, 'with a probability of ', round(proba[0][0] * 100, 2), "%")
    return proba



def myresult(request):
    print(12)
    p1=request.POST['player1']
    p2=request.POST['player2']

    import pandas as pd
    import numpy as np
    import re
    df = pd.read_csv('analysis/templates/analysis/data.csv')
    b_age = df['B_age']  # we replace B_age to put it among B features
    df.drop(['B_age'], axis=1, inplace=True)
    df.insert(76, "B_age", b_age)
    df_fe = df.copy()
    last_fight = df.loc[0, ['date']]
    limit_date = '2001-04-01'
    df = df[(df['date'] > limit_date)]
    na = []
    for index, col in enumerate(df):
        na.append((index, df[col].isna().sum()))
    na_sorted = na.copy()
    na_sorted.sort(key=lambda x: x[1], reverse=True)
    from sklearn.impute import SimpleImputer

    imp_features = ['R_Weight_lbs', 'R_Height_cms', 'B_Height_cms', 'R_age', 'B_age', 'R_Reach_cms', 'B_Reach_cms']
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

    for feature in imp_features:
        imp_feature = imp_median.fit_transform(df[feature].values.reshape(-1, 1))
        df[feature] = imp_feature

    imp_stance = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_R_stance = imp_stance.fit_transform(df['R_Stance'].values.reshape(-1, 1))
    imp_B_stance = imp_stance.fit_transform(df['B_Stance'].values.reshape(-1, 1))
    df['R_Stance'] = imp_R_stance
    df['B_Stance'] = imp_B_stance
    na_features = ['B_avg_BODY_att', 'R_avg_BODY_att']
    df.dropna(subset=na_features, inplace=True)

    df.drop(['Referee', 'location'], axis=1, inplace=True)
    df.drop(['B_draw', 'R_draw'], axis=1, inplace=True)
    df = df[df['Winner'] != 'Draw']
    df = df[df['weight_class'] != 'Catch Weight']
    fighters = list_fighters(df, '2017-01-01')
    df_train = build_df(df, fighters, 0)
    df_test = build_df(df, fighters, 1)
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from sklearn.compose import make_column_transformer

    preprocessor = make_column_transformer((OrdinalEncoder(), ['weight_class', 'B_Stance', 'R_Stance']),
                                           remainder='passthrough')

    # If the winner is from the Red corner, Winner label will be encoded as 1, otherwise it will be 0 (Blue corner)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['Winner'])
    y_test = label_encoder.transform(df_test['Winner'])

    X_train, X_test = df_train.drop(['Winner'], axis=1), df_test.drop(['Winner'], axis=1)
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score

    # Random Forest composed of 100 decision trees. We optimized parameters using cross-validation and GridSearch tool paired together
    random_forest = RandomForestClassifier(n_estimators=100,
                                           criterion='entropy',
                                           max_depth=10,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           random_state=0)

    model = Pipeline([('encoding', preprocessor), ('random_forest', random_forest)])
    model.fit(X_train, y_train)
    result = predict(df, model, p1, p2, 'Welterweight', 5, True)
    result1 = result.tolist()
    result2 = result1[0]
    context={}
    context['player1']=p1
    context['player2']=p2
    probability_p1=round(result2[0]*100,1)
    probability_p2=round(result2[1]*100,1)
    wintype=""
    myround=1
    if result2[0]>result2[1]:
        winner=p1
        if result2[0]<70:
            wintype = "Technical Knockout"
            if result2[0] - result2[1] <= 3:
                myround = 5
            elif result2[0] - result2[1] >= 4:
                myround = 4
            elif result2[0] - result2[1] <= 7:
                myround = 3
            elif result2[0] - result2[1] <= 9:
                myround = 2
            elif result2[0] - result2[1] <= 11:
                myround = 1


        else:
            wintype="Technical Knockout"
    else:
        winner=p2
        if result2[1]<70:
            wintype="Knockout"
            wintype = "Technical Knockout"
            if result2[0] - result2[1] <= 3:
                myround = 5
            elif result2[0] - result2[1] <= 5:
                myround = 4
            elif result2[0] - result2[1] <= 7:
                myround = 3
            elif result2[0] - result2[1] <= 9:
                myround = 2
            elif result2[0] - result2[1] <= 11:
                myround = 1
        else:
            wintype="Technical Knockout"
    context['winner']=winner
    context['p1prob']=probability_p1
    context['p2prob']=probability_p2
    context['wintype']=wintype
    context['round']=myround

    return render(request, "analysis/dataset2.html",context)





def data_extraction_form(request):
    if 'view' in request.POST:
        from .models import features
        temp = features.objects.all()
        features = list(temp.values_list('feature', flat=True))
        myfeature = features[-1]
        print(myfeature)
        static_path = "analysis/templates/analysis/"
        if myfeature == "Bag Of Words":
            filename = "Bagofwords.csv"
        elif myfeature == "Part of Speech Tagging":
            filename = "POS.csv"
        elif myfeature == "TF-IDF":
            filename = "tf_idf.csv"
        elif myfeature == "Discrete Positive":
            filename = "Discrete_positive.csv"
        elif myfeature == "Discrete negative":
            filename = "Discrete_negative.csv"
        elif myfeature == "Polarity":
            filename = "Polarity.csv"
        elif myfeature == "Sentiments":
            filename = "Sentiment.csv"
        elif myfeature == "All":
            filename = "all.csv"

        df = pd.read_csv(static_path + filename)
        return HttpResponse(df.to_html())

    if 'gonext' in request.POST:
        selected = request.POST['prep']
        print(selected)
        result = preprocessing(prep=selected)
        result.save()
        return render(request, 'analysis/classifier.html')


#
# def data_extraction_form(request):
#     selected = request.POST['prep']
#     print(selected)
#     result=preprocessing(prep=selected)
#     result.save()
#     return render(request, 'analysis/classifier.html')


# handling feature selection form

def feature_selection_form(request):
    from .models import features
    selected = request.POST['feature']
    result = features(feature=selected)
    result.save()
    from .models import features
    temp = features.objects.all()
    features = list(temp.values_list('feature', flat=True))
    myfeature = features[-1]
    print(myfeature)
    context={}
    context['test']=myfeature
    return render(request, 'analysis/data_preprocessing.html',context)

    # if 'abc' in request.POST:
    #     return render(request, 'analysis/about.html')
    #
    # if 'myfeature' in request.POST:
    #     selected=request.POST['feature']
    #     result = features(feature=selected)
    #     result.save()
    #     return render(request,'analysis/data_preprocessing.html')


from random import randint as ml_metric
from django.shortcuts import redirect
# def index(request):
#     return HttpResponse("<h1>Hello everyone</h1>")
from django.contrib import messages


def index(request):
    return render(request, "analysis/login.html")

def index2(request):
    return render(request, "analysis/index.html")

def homepage(request):
    return render(request, "analysis/index.html")


def login_check(request):
    input_email = request.POST['email']
    input_password = request.POST['password']
    from .models import registration

    temp = registration.objects.all()
    emails = list(temp.values_list('email', flat=True))
    passwords = list(temp.values_list('password', flat=True))

    e = False
    p = False
    for item in emails:
        print(item)
    if input_email in emails:
        e = True
    if input_password in passwords:
        p = True

    if e == True and p == True:
        myobj = registration.objects.get(email=input_email)
        value_of_name = myobj.name
        context = {}
        context['name'] = value_of_name
        return render(request, "analysis/index.html", context)

    else:
        print('failed')
        return render(request, "analysis/login3.html")


def about(request):
    return render(request, "analysis/about.html")


def login(request):
    return render(request, "analysis/login.html")


def dataset_2(request):
    return render(request, "analysis/dataset2.html")


def dp(request):
    return render(request, "analysis/dp2.html")


from itertools import chain
from functools import reduce


def history(request):
    ml = ml_model.objects.all()
    prep = preprocessing.objects.all()
    feat = features.objects.all()
    from analysis.models import registration as regis
    reg = regis.objects.all()
    mydata = list(zip(ml, prep, feat))

    # mydata = list(chain(ml,prep,feat,reg))

    return render(request, 'analysis/history.html', {'dataset': mydata})


def delrecord(request):
    from .models import ml_model, features, preprocessing
    myid = request.POST['prodId']
    ml_model.objects.filter(id=myid).delete()
    features.objects.filter(id=myid).delete()
    preprocessing.objects.filter(id=myid).delete()
    ml = ml_model.objects.all()
    prep = preprocessing.objects.all()
    feat = features.objects.all()
    from analysis.models import registration as regis
    reg = regis.objects.all()
    mydata = list(zip(ml, prep, feat))

    return render(request, 'analysis/history.html', {'dataset': mydata})


def registration(request):
    return render(request, "analysis/registration.html")


def registration2(request):
    name = request.POST['name']
    email = request.POST['email2']
    username = request.POST['username']
    password = request.POST['password']
    from .models import registration
    obj = registration(name=name, email=email, username=username, password=password)
    obj.save()
    return render(request, "analysis/login2.html")


def error(request):
    return render(request, "analysis/error.html")


def view_dataset_(request):
    df = pd.read_csv("analysis/templates/analysis/shopify_main_data.csv")
    geeks_object = df.to_html()
    return HttpResponse(geeks_object)


def dataset(request):
    return render(request, "analysis/dataset.html")


def algorithm_selection(request):
    return render(request, "analysis/algorithm_selection.html")


def algorithm_evaluation(request):
    return render(request, "analysis/algorithm_evaluation.html")


def contact_us(request):
    return render(request, "analysis/contact_us.html")


def data_selection(request):
    return render(request, "analysis/data_selection.html")


def feature_selection(request):
    return render(request, "analysis/feature_selection.html")


def classifier(request):
    return render(request, "analysis/classifier.html")


def unseen_review(request):
    return render(request, "analysis/unseen_review.html")


import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

import string

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes",
                'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']
STOPWORDS = set(stopwordlist)

english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', data)


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


from nltk.tokenize import RegexpTokenizer
import nltk

st = nltk.PorterStemmer()


def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data


lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data


from nltk.tokenize import RegexpTokenizer


def test_data(sentence):
    DATASET_COLUMNS = ['app_id', 'author', 'rating', 'posted_at', 'body', 'helpful_count', 'label']
    DATASET_ENCODING = "ISO-8859-1"
    df = pd.read_csv('analysis/templates/shopify_main_data.csv', encoding=DATASET_ENCODING)
    text, sentiment = list(df['body']), list(df['Label'])
    data = df[['body', 'Label']]
    data['Label'] = data['Label'].replace(4, 1)
    data_pos = data[data['Label'] == "Happy"]
    data_neg = data[data['Label'] == "Unhappy"]
    data_pos = data_pos.iloc[:int(12000)]
    data_neg = data_neg.iloc[:int(12000)]
    dataset = pd.concat([data_pos, data_neg])
    dataset['body'] = dataset['body'].str.lower()
    dataset['body'] = dataset['body'].apply(lambda text: cleaning_stopwords(text))
    dataset['body'] = dataset['body'].apply(lambda x: cleaning_punctuations(x))
    dataset['body'] = dataset['body'].apply(lambda x: cleaning_repeating_char(x))
    dataset['body'] = dataset['body'].apply(lambda x: cleaning_URLs(x))
    dataset['body'] = dataset['body'].apply(lambda x: cleaning_numbers(x))
    tokenizer = RegexpTokenizer(r'w+')
    dataset['body'] = dataset['body'].apply(tokenizer.tokenize)
    dataset['body'] = dataset['body'].apply(lambda x: stemming_on_text(x))
    dataset['body'] = dataset['body'].apply(lambda x: lemmatizer_on_text(x))
    X = data.body
    y = data.Label
    data_neg = data['body'][:800000]
    data_pos = data['body'][10000:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26105111)
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    RMmodel = RandomForestClassifier()
    RMmodel.fit(X_train, y_train)
    y_pred = RMmodel.predict(X_test)
    a = classification_report(y_test, y_pred)
    b = a.split()
    precision_of_happy = (float(b[5]))
    precision_of_unhappy = (float(b[10]))
    positive_words = ['happy', 'good', 'nice', 'great', 'wonderful', 'beautiful', 'charming', 'perfect','glad','cool']
    tokens = sentence.split()
    mypred = ""

    if (precision_of_happy > precision_of_unhappy):
        prediction = "Happy"
    else:
        for item in positive_words:
            if item in tokens:
                mypred = "Happy"
                break
            else:
                mypred = "Unhappy"

    return mypred


def results(request):
    context = {}
    system = request.POST.get("text")
    context['system'] = system
    sentence = context['system']
    print(sentence)
    context['prediction'] = test_data(sentence)
    from .models import ml_model
    o1 = ml_model.objects.all()
    a = list(o1.values_list('accuracy', flat=True))
    p = list(o1.values_list('precision', flat=True))
    f = list(o1.values_list('fmeasure', flat=True))
    r = list(o1.values_list('recall', flat=True))
    a_=a[-1]
    p_=p[-1]
    f_=f[-1]
    r_=r[-1]
    context['a']=a_
    context['p']=p_
    context['f']=f_
    context['r']=r_

    # return HttpResponse(f"<h1>{context['system']}</h1>")
    return render(request, "analysis/result.html", context)


def mail(request):
    print('pakistan')
    from django.core.mail import EmailMessage
    myemail=request.POST['emailid']
    text=request.POST['text']
    query=request.POST['query']
    email = EmailMessage(text, query,'or34666@gmail.com' ,to=[myemail])
    email.send()
    return render(request, "analysis/contact_us2.html")
