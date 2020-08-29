import os
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from urllib.parse import urlparse
from datetime import timedelta
from urlextract import URLExtract
from keras.preprocessing.text import Tokenizer

extractor = URLExtract()
one_hour = timedelta(hours=1)

bio_tokenizer, website_tokenizer, email_tokenizer = None, None, None

def load_tokenizers():
    global bio_tokenizer, website_tokenizer, email_tokenizer

    with open("../data/bio_tokenizer.pickle", "rb") as f:
        bio_tokenizer = pickle.load(f)

    with open("../data/website_tokenizer.pickle", "rb") as f:
        website_tokenizer = pickle.load(f)

    with open("../data/email_tokenizer.pickle", "rb") as f:
        email_tokenizer = pickle.load(f)

# editor preprocessing

extractor = URLExtract()
one_hour = timedelta(hours=1)


def preprocess_editor(editor, spam=1):
    load_tokenizers()
    # Apparently there are users with unset member_since
    if editor["member_since"] is not None:
        # These shouldn't be none but you can't trust the database
        if editor["last_updated"] is not None:
            update_delta = (editor["last_updated"] - editor["member_since"]) / one_hour
        else:
            update_delta = -1

        if editor["last_login_date"] is not None:
            login_delta = (editor["last_login_date"] - editor["member_since"]) / one_hour
        else:
            login_delta = -1

        # Confirm date may be None
        if editor["email_confirm_date"] is not None:
            conf_delta = (editor["email_confirm_date"] - editor["member_since"]) / one_hour
        else:
            conf_delta = -1
    else:
        update_delta, login_delta, conf_delta = -2, -2, -2

    # Email domain
    email_domain = email_tokenizer.texts_to_sequences([editor["email"].split("@")[1]])[0]
    if len(email_domain) == 0:
        email_token = 1024
    else:
        email_token = email_domain[0]

    # Website domain
    domain = urlparse(editor["website"]).hostname
    if domain is not None:
        website_domain = website_tokenizer.texts_to_sequences(urlparse(editor["website"]).hostname)[0]
        if len(website_domain) == 0:
            website_token = 1023
        else:
            website_token = email_domain[0]
    else:
        website_token = 1024

    # Bio metadata
    if editor["bio"] is not None:
        bio_len = len(editor["bio"])
        bio_urls = extractor.has_urls(editor["bio"])
        bio = bio_tokenizer.texts_to_matrix([editor["bio"]], mode="tfidf")[0]
    else:
        bio_len, bio_urls = 0, 0
        bio = np.zeros(512)

    data = np.array([
        spam, # spam classification
        editor["area"] is not None, # Area Set
        editor["gender"] is not None, # Gender
        editor["birth_date"] is not None, # Birth date set
        editor["privs"] != 0, # Nonzero privs
        bio_len, # Bio length
        bio_urls, # URLs in bio
        conf_delta, # Confirmation delta
        update_delta, # Last updated delta
        login_delta, # Last login delta
        email_token, # Email domain
        website_token, # Website domain
    ], dtype=np.float32)

    data = np.concatenate((data, bio))
    return data