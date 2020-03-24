#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import os
import zipfile
import datetime
import xgboost as xgb


# %% [markdown]
#  ## Load Data

# %%
def read_data(path, nrows, **kwargs):
    columns=['bert','hashtags','tweet_id','media','links','domains','type','language',
              'timestamp','EE_user_id','EE_follower_count','EE_following_count',
              'EE_verified','EE_account_creation','ER_user_id','ER_follower_count',
              'ER_following_count','ER_verified','ER_account_creation',
              'engagee_follows_engager','reply_timestamp','retweet_timestamp',
               'quote_timestamp','like_timestamp']
    def dateparse (time_in_secs):   
        try:
            return datetime.datetime.fromtimestamp(float(time_in_secs))
        except:
            return np.nan
    parse_dates = [c for c in columns if 'timestamp' in c or 'creation' in c]
    return pd.read_csv(path, nrows=nrows, sep='\x01', names=columns, parse_dates=parse_dates, date_parser=dateparse, **kwargs).set_index('tweet_id')


# ### Train Data

# %%
# training = read_data('training.tsv',skiprows=int(2e6), nrows=int(1e6)).to_pickle('training_1e6_2.pkl')


# %%
training = pd.read_pickle('training_1e6_0.pkl')


# %%
def get_features(df, model):
    return df[['EE_follower_count', 'EE_following_count','ER_verified', 'ER_follower_count', 'ER_following_count']]


# %%
def get_y(training, model):
    if model=='any':
        return training[['reply_timestamp','retweet_timestamp','quote_timestamp','like_timestamp']].notna().apply(lambda x: np.any(x), axis=1)
    # model reply, retweet, quote, reply
    return (-training['{}_timestamp'.format(model)].isnull()).astype(int)


# %%

# %% [markdown]
# ## Model

# %% [markdown]
# ### Global

# %%
X_train_any, X_test_any, y_train_any, y_test_any =  train_test_split(get_features(training, 'any'), 
                                                     get_y(training, 'any'), 
                                                     test_size=0.4, 
                                                     random_state=0)


# %%
model_any = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_any.fit(X_train_any, y_train_any)


# %%
model_any.score(X_train_any, y_train_any)

# %% [markdown]
# ### Retweet

# %%
X_train_rt, X_test_rt, y_train_rt, y_test_rt =  train_test_split(get_features(training, 'retweet'), 
                                                     get_y(training, 'retweet'), 
                                                     test_size=0.4, 
                                                     random_state=0)


# %%
model_rt = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_rt.fit(X_train_rt, y_train_rt)


# %%
# cross_val_score(model_rt, X_test, y_test, cv=3)    
model_rt.score(X_test_rt, y_test_rt)

# %% [markdown]
# ### Reply

# %%
X_train_rp, X_test_rp, y_train_rp, y_test_rp =  train_test_split(get_features(training, 'reply'), 
                                                     get_y(training, 'reply'),
                                                     test_size=0.4, 
                                                     random_state=0)


# %%
model_rp = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_rp.fit(X_train_rp, y_train_rp)


# %%
model_rp.score(X_test_rp, y_test_rp)

# %% [markdown]
# ### Like

# %%
X_train_like, X_test_like, y_train_like, y_test_like =  train_test_split(get_features(training, 'like'), 
                                                     get_y(training, 'like'),
                                                     test_size=0.4, 
                                                     random_state=0)


# %%
model_like = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_like.fit(X_train_like, y_train_like)


# %%
model_like.score(X_test_like, y_test_like) 

# %% [markdown]
# ### Quote

# %%
X_train_q, X_test_q, y_train_q, y_test_q =  train_test_split(get_features(training, 'quote'), 
                                                     get_y(training, 'quote'),
                                                     test_size=0.4, 
                                                     random_state=0)


# %%
model_quote = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_quote.fit(X_train_q, y_train_q)


# %%
model_like.score(X_test_q, y_test_q)

# %% [markdown]
# ## Validation & submission

# %%
# validation = read_data('val.tsv', skiprows=200000, nrows=100000)
# validation.to_pickle('validation_1e5_2.pkl')


# %%
validation = pd.read_pickle('validation_1e5_0.pkl')

# %%
validation.iloc[0].name

# %%
rt = model_rt.predict(get_features(validation, 'retweet'))
rp = model_rp.predict(get_features(validation, 'reply'))
like = model_like.predict(get_features(validation, 'like'))
quote = model_like.predict(get_features(validation, 'quote'))

# %%
submission = pd.DataFrame([rt,rp,like, quote], columns=validation.index, index=['retweet', 'reply', 'like', 'quote']).transpose()


# %%
submission.index.name='tweet_id'


# %%
submission['user_id']=validation['ER_user_id']


# %%
submission.head()

