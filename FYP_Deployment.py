import streamlit as st
#data pre-processing
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import emoji
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

st.title("Expose Cyberbully tweets on Twitter using Machine Learning")
abstract = st.expander("Abstract")
if abstract:
    abstract.write("The advancements of technology along with the digitization of the relationships made a great impact among the centennials to mandatorily maintain a social media account. Despite the entertainment that social media provides, cyberbullying has been identified as a real issue in Malaysia where these centennials are victims. However, a smaller number of studies have been reported in this regard in terms of detecting the attempt of cyberbullying on social media. On this background, a solution using suitable data science techniques which can help to detect the attempt of cyberbullying on social media would be ideal. This research proposed to use suspicious tweets dataset from Kaggle to train three classifiers for supervised learning using Naïve Bayes, SVM, and LSTM. Model tuning was performed using Random Grid Search and Keras tuner. Overall, the model had an accuracy rate of 88% indicating that the optimization tuning functioned properly. Out of the three, Naïve Bayes performed the best in terms of both accuracy and area under the curve (AUC) values with 88.4% and 0.81 respectively.")

about = st.expander("About")
if about:
    about.write("The application below demonstrates a machine learning learning model (Naïve Bayes) that has been trained to detect cyberbullying in tweets from Twitter.")
    about.markdown("**Information on the Classifier**")
    if about.checkbox("About Classifer"):
        about.markdown('**Model:** Naïve Bayes')
        about.markdown('**Vectorizer:** Count')
        about.markdown('**Test-Train splitting:** 20% - 80%')
        about.markdown('**Lemmetization/Stemmer:** Wordnet with POS tagging')
        
    if about.checkbox("Evaluation Results"):
        about.markdown('**Accuracy:** 88%')
        about.markdown('**Precision:** 91%')
        about.markdown('**Recall:** 88%')
        about.markdown('**F1 Score:** 89%')
        about.markdown('**AUC Score:** 0.81')

related = st.expander("Related Links")
if related:
    related.write("[Dataset](https://www.kaggle.com/syedabbasraza/suspicious-tweets)")

temp = st.text_area("Insert Tweet:")
btn_analyse = st.button('Analyse')

if btn_analyse:
    def get_wordnet_pos(word):
    
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    temp = temp.lower()
    temp = emoji.demojize(temp)
    temp = nltk.word_tokenize(temp)

    #remove punctuations
    temp = [i for i in temp if i not in set(string.punctuation)]

    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    temp = [word for word in temp if word not in english_stops]
    temp = [word for word in temp if word not in set(characters_to_remove)]
    
    #Lemmatize with POS Tagging
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    temp = [wordnet_lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in temp]

    def eval_avg(eval):
        return sum(eval)/len(eval)

    temp = [str (item) for item in temp]
    vectorizer = pickle.load(open("model_for_deployment/Vectorize_Save", 'rb'))
    vect = vectorizer.transform(temp)

    #Load Naive Bayes Model
    load_nb = pickle.load(open("model_for_deployment/NaiveBayes_Model.sav", 'rb'))

    #Predict using Naive Bayes
    nb_pred = load_nb.predict(vect)
    print(nb_pred)

    result_nb = [int (item) for item in nb_pred]
    result_nb = eval_avg(result_nb)

    print(temp)
    print(result_nb)

    output_nb = (result_nb > 0.5)
    print(output_nb)

    st.markdown("Naive Bayes Prediction:")
    if (output_nb == 1):
        st.success("No Cyberbully intent detected")
    else:
        st.error("Possible Cyberbully intent!")
        help = st.expander("Need Help?")
        if help:
            help.write("[R.AGE Website](https://www.rage.com.my/helplines-and-counselling/)")
            help.write("Call the national 24-hour hotline **15999** to call to report abuse, bullying, neglect, etc.")
else:
    st.warning("Please insert the tweet!")
  
