from django.shortcuts import render
from django.http import HttpResponse, Http404
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import permissions
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from spacy.lang.en import English
import re
import enchant
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import string
from nltk.corpus import stopwords
from sklearn import preprocessing
from nltk.util import pr
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from django.conf import settings


# Create your views here.
def index(request):
    return render(request, "singlepage/index.html")


# texts = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam tortor mauris, maximus semper volutpat vitae, varius placerat dui. Nunc consequat dictum est, at vestibulum est hendrerit at. Mauris suscipit neque ultrices nisl interdum accumsan. Sed euismod, ligula eget tristique semper, lecleo mi nec orci. Curabitur hendrerit, est in ",
#         "Praesent euismod auctor quam, id congue tellus malesuada vitae. Ut sed lacinia quam. Sed vitae mattis metus, vel gravida ante. Praesent tincidunt nulla non sapien tincidunt, vitae semper diam faucibus. Nulla venenatis tincidunt efficitur. Integer justo nunc, egestas eget dignissim dignissim,  facilisis, dictum nunc ut, tincidunt diam.",
#         "Morbi imperdiet nunc ac quam hendrerit faucibus. Morbi viverra justo est, ut bibendum lacus vehicula at. Fusce eget risus arcu. Quisque dictum porttitor nisl, eget condimentum leo mollis sed. Proin justo nisl, lacinia id erat in, suscipit ultrices nisi. Suspendisse placerat nulla at volutpat ultricies"]


# def section(request, num):
#     if 1 <= num <= 3:
#         return HttpResponse(texts[num-1])
#     else:
#         raise Http404("No such section")


def counter(request):
    text = request.data.get('summary', None)
    amount = len(text.split())
    return render(request,'index.html',{'amount' : amount})


class ReturnSummary(APIView):
    cv = settings.CV
    clf = settings.CLF
    # permissions = (permissions.AllowAny,)



    def hate_speech_detection(self, user):
        if len(user) < 1:
            return None
        else:
            data = self.cv.transform([user]).toarray()
            a = self.clf.predict(data)
#         y_test = label_encoder.inverse_transform(y_test)
#         y_pred_rfc = label_encoder.inverse_transform(a)
#         two_d_compare(y_test,a,model_name)
            if a:
                return a[0]

    def make_summary(self, text,max_sent_in_summary ):
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        nlp = English()
        nlp.create_pipe('sentencizer')
        nlp.add_pipe('sentencizer')
        doc = nlp(text.replace("\n", ""))
        sentences = [sent.text.strip() for sent in doc.sents]
#     print(sentences)
        sentence_organizer = {k:v for v,k in enumerate(sentences)}
#     print(sentence_organizer)
        tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3), 
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words = 'english')
        tf_idf_vectorizer.fit(sentences)
        sentence_vectors = tf_idf_vectorizer.transform(sentences)
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
        # print(sentence_scores)
        N = max_sent_in_summary
        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
        # print(top_n_sentences[0])
        mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
        summary = " ".join(ordered_scored_sentences)
        return summary

    def post(self, request):
        # print(request.data)
        # if request.FILES.get('upload_file', None) and request.FILES.get('upload_file').name.endswith('.txt'):
        if request.data.get('summary', None) and len(request.data.get('summary', None).split()) >= 500:
            text = request.data.get('summary')
            # print(text)
            number = 4
            if (len(text.split())) > 1000:
                number = 6
            elif(len(text.split())) > 1500:
                number = 8
            seen = []
            answer = []
            for line in text.split('.'):
                line = line.strip()
                if line not in seen:
                    seen.append(line)
                    answer.append(line)
            text = '.'.join(answer)
            if(len(text.split()) >= 500):
                summary = self.make_summary(text,number)
                summary1 = summary
                stop_words = set(stopwords.words('english'))
                word_tokens = word_tokenize(summary1)
                summary1 = [w for w in word_tokens if not w.lower() in stop_words]
                summary1 = []
                for w in word_tokens:
                    if w not in stop_words:
                        summary1.append(w)
                summary1 = ' '.join(map(str, summary1))
                spell = Speller(lang='en')
                ab = settings.DICWORD
                mistakes = [(x) for x in summary1.split() if x != spell(x) and x not in ab]
            #   print(mistakes)
            #   summary = " ".join([re.sub(r"[.:-]+",'',x) for x in summary.split() if x != "."])
                # print(summary)
            #   summary = " ".join([re.sub(r'[^\w\s]','',x) for x in summary.split() if x != "."])
                for i in summary.split():
#               print(i,j)
                    for j in mistakes:
                        if i == j:
                            summary = summary.split()
                            undlined = ''
                            for char in i:
                                undlined += char + "\u0332"
                            CRED = """<font style="background-color: orange;">"""
                            # <h2 style="background-color: steelblue;">
                            CEND = "</font>"
                            undlined = CRED + undlined + CEND
                            summary = list(map(lambda x: x.replace(i, undlined), summary))
                            summary = ' '.join(map(str, summary))
                tagged_summary = [(x.strip(),self.hate_speech_detection(user=x.strip())) for x in summary.split() if x]
                # print(tagged_summary)
                final_summary = ""
                for i in tagged_summary:
                    if i[1] == 'Hate Speech':
                        CRED = """<font style="background-color: Red;">"""
                        CEND = "</font>"
                    # CRED = '\033[91m'
                    # CEND = '\033[0m'
                        final_summary += CRED + i[0] + CEND +" "
                        # print(i)
                    else:
                        final_summary += i[0]+" "
                # final_summary += "."
                # print(repr(final_summary))
                resp = {"summary":final_summary}
                return Response(resp,status = 200)
            else:
                errorr = "Text is less than 500 Words After Removing dupilcate sentences"
                resp1 = {"summary":errorr}
                return Response(resp1,status = 200)
        else:
            error = "Text is not enough for Summary"
            resp2 ={"summary":error}
            return Response(resp2,status = 200)
