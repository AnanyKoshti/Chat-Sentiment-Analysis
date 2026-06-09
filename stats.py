from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd 
from collections import Counter


#Object for Url extraction
extract = URLExtract()

#functions for stats fetching
def fetch_stats(selected_user, df):
    if selected_user != 'All Users':
        
        df = df[df['user'] == selected_user]#fetching number of messages
    number_of_messages = df.shape[0]
    
    number_of_words = []# fetching number of words
    for message in df['message']:
        number_of_words.extend(message.split())
        
    media_shared = df[df['message'] == '<Media omitted>\n'].shape[0]#Total media shared
    
    links = []# Total links shared
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    return number_of_messages, len(number_of_words), media_shared, len(links)


def Active_user(df):

    if df.empty:
        return pd.Series(dtype=int), pd.DataFrame()

    act_user = df['user'].value_counts().head()

    dfpercent = (
        round((df['user'].value_counts() / df.shape[0]) * 100, 2)
        .reset_index()
    )

    dfpercent.columns = ['Name', 'Percent']

    return act_user, dfpercent


#function to create wordcloud

def cloud_of_words(selected_user, df):

    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    text = df['message'].dropna().astype(str).str.cat(sep=' ')

    if len(text.strip()) == 0:
        return None

    wcloud = WordCloud(
        width=350,
        height=250,
        min_font_size=8,
        background_color='white'
    )

    return wcloud.generate(text)


#Common words generally used in chat
def common_words(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
    
    rm_grp_noti = df[df['user'] != 'Group_Message']# Removing group notification
    rm_media_omi = rm_grp_noti[rm_grp_noti['message']!= '<Media omitted>\n']# removing <Media omitted>
    
    with open('stopwords_hinglish.txt', 'r', encoding='utf-8') as f:
       stop_words = set(f.read().splitlines())
    message_words = []
    for message in rm_media_omi['message']:
        for word in message.lower().split():
            if word not in stop_words:
                message_words.append(word)
                
    if len(message_words) == 0:
        return pd.DataFrame(columns=[0, 1])

    df_common = pd.DataFrame(
        Counter(message_words).most_common(20)
    )

    return df_common

#Monthly messages from single user or Overall

def monthly_usage(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    year_ana = df.groupby(['year','month_num','month']).count()['message'].reset_index()
    
    month_year = []
    for i in range(year_ana.shape[0]):
        month_year.append(
            str(year_ana.iloc[i]['month']) +
            '-' +
            str(year_ana.iloc[i]['year'])
        )
    year_ana['Time'] = month_year
    return year_ana

# Daily usage by users
def daily_msgs(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    daily_usage = df.groupby('daily').count()['message'].reset_index()
    return daily_usage

# Active users on a particular day in a week
def weekly_activity(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    return df['day_name'].value_counts()

# Active users on a particular month

def month_avtivity(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    return df['month'].value_counts()


#######SENTIMENT Analysis for the particular user or overall#####

def sentiment_func(selected_user, df):

    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    required_cols = ['neg', 'neu', 'pos', 'compound']

    for col in required_cols:
        if col not in df.columns:
            return [0, 0, 0, 0]

    neg_total = df['neg'].fillna(0).sum()
    neu_total = df['neu'].fillna(0).sum()
    pos_total = df['pos'].fillna(0).sum()
    comp_total = abs(df['compound'].fillna(0).sum())

    return [neg_total, neu_total, pos_total, comp_total]


def sentiment_neg_pos(selected_user, df):

    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    if df.empty:
        return 0, 0

    scr_count = df['compare_score'].value_counts()

    pos_count = scr_count.get('Positive', 0)
    neg_count = scr_count.get('Negative', 0)

    return pos_count, neg_count
    

    
    
    
    
    

            
