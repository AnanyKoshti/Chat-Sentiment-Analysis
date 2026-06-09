import streamlit as st 
import analysis
import stats
import matplotlib.pyplot as plt 


#############################______________FRONTEND_________________#####################
st.sidebar.title("Analyze chat!")

st.title("Whatsapp Chats Sentiment Analysis")
st.subheader("How to use?")
st.text("Step1:Just open your whatsapp!")
st.text("Step2:Export chats (without media)* ")
st.text("Step3:Upload .txt file on your PC")
st.text("Step4:Browse that file here only!")
st.text("Step5:Select All Users or a particular User!")
st.text("Step6:Hit the button \"Analyze Now!\"")

col1,col2 = st.columns(2)
with col1:
    st.subheader("Capability")
    st.text("The capability of a sentiment analysis model")
    st.text("in English and Hinglish language is to")
    st.text("accurately classify text into positive,")
    st.text("negative, or neutral sentiment categories.")
    
    
with col2:
    st.subheader("Limitations")
    st.text("Chats must be in the supported format.")
    st.text("May not work when Key Error! occured, due to")
    st.text("absence of req* field for particular user.")
    
st.markdown("**:green[MORE CHATS MORE EFFICIENT RESULTS!]**")


upload_file = st.sidebar.file_uploader("Choose a file")
if upload_file is not None:
    data_bytes = upload_file.getvalue()
    data = data_bytes.decode('latin-1')
    df = analysis.Analyse_Data(data)
    st.header("ALL CHATS")
    st.dataframe(df)
    
    #finding unique users
    userlist = df['user'].unique().tolist()
    
    if 'Group_Message' in userlist:
        userlist.remove('Group_Message')
        
    userlist.sort()
    userlist.insert(0,"All Users")
    
    

    #Select user from sidebar
    selected_user = st.sidebar.selectbox("Show Analysis",userlist)
    
    #Button for selecting user
    if st.sidebar.button("Analyze Now!"):
        
        number_of_messages, number_of_words, media_shared, links = stats.fetch_stats(selected_user, df)
        
        
        # Presenting TOTAL MESSAGES, TOTAL MESSAGES, TOTAL MEDIA, TOTAL LINKS
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.header("TOTAL MESSAGES")
            st.title(number_of_messages)
            
        with col2:
            st.header("TOTAL WORDS")
            st.title(number_of_words)
            
        with col3:
            st.header("TOTAL MEDIA")
            st.title(media_shared)
            
        with col4:
            st.header("TOTAL LINKS")
            st.title(links)
            
        
            
            
        # Evaluating Active user only applicable for Group
        if selected_user == 'All Users':
            st.title("ACTIVE USERS & USAGE")
            act_user, dfpercent = stats.Active_user(df)
            fig , ax = plt.subplots()
            
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(act_user.index, act_user.values, color = 'green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
                
            with col2:
                st.dataframe(dfpercent)
                
                
        # Positive VS Negative messages
        pos_count, neg_count = stats.sentiment_neg_pos(selected_user,df)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Chats Monitoring")
            st.text("If the compound score is less than zero, the")
            st.text("sentiment is considered negative. Conversely,")
            st.text("if the compound score is greater than zero,")
            st.text("the sentiment is considered positive. The closer")
            st.text("ranges from -1 to 1, to quantify the sentiment of the text.")
            
            
            
        with col2:
            st.header("Positive Vs Negative Messages")
            labs = ['Positive', 'Negative']
            pie_sizes = [pos_count, neg_count]
            explode = (0.1,0.1)
            fig, ax = plt.subplots()
            
            if sum(pie_sizes) > 0:
                ax.pie(
                    pie_sizes,
                    explode=explode,
                    labels=labs,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("No sentiment data available.")
            
                
        
        # Crearing wordcloud using messages
        df_wcloud = stats.cloud_of_words(selected_user,df)
        
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(df_wcloud)
            st.header("Message Cloud")
            st.pyplot(fig)
        
        # Showing sentiment Analysis chart
        df_sentiment = stats.sentiment_func(selected_user,df)
                
        with col2:
            labels = ['Negative', 'Neutral', 'Positive', 'Compound']
            explode = (0.1,0.1,0.1,0.1)
            fig, ax = plt.subplots()
            ax.pie(df_sentiment, explode=explode, labels=labels, autopct= '%1.1f%%',startangle=90)
            ax.axis('equal')
            st.header("Sentiment Distribution")
            st.pyplot(fig)
                
                
        # Common words used in chats regularly
        df_common = stats.common_words(selected_user,df)
        st.header("Common Words Used")
        fig,ax = plt.subplots()
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_common)
        
        with col2:
            ax.barh(df_common[0],df_common[1],color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            
            
        # To show messeges sent by each user and overall users monthly and year wise
        year_ana = stats.monthly_usage(selected_user,df)
        fig,ax = plt.subplots()
        
        col1, col2 = st.columns(2)
        with col1:
            ax.plot(year_ana['Time'], year_ana['message'],color='orange')
            plt.xticks(rotation='vertical')
            st.header("Monthly Usage")
            st.pyplot(fig)
        
        # To show daily messages sent by users
        daily_usage = stats.daily_msgs(selected_user,df)    
        with col2:
            fig,ax = plt.subplots()
            ax.plot(daily_usage['daily'], daily_usage['message'],color='green')
            plt.xticks(rotation='vertical')
            st.header("Daily Usage")
            st.pyplot(fig)
            
        # Weekly day wise activity, active users on particular day
        col1,col2 = st.columns(2)
        
        with col1:
            st.header("Active Days")
            active_day = stats.weekly_activity(selected_user,df)
            fig, ax = plt.subplots()
            plt.xticks(rotation='vertical')
            ax.bar(active_day.index,active_day.values,color='red')
            st.pyplot(fig)
            
        # Monthly active users for the particular month
        with col2:
            st.header("Active Months")
            active_month = stats.month_avtivity(selected_user,df)
            fig, ax = plt.subplots()
            plt.xticks(rotation='vertical')
            ax.bar(active_month.index,active_month.values,color='blue')
            st.pyplot(fig)
         
        st.subheader("Remember:")    
        st.text("*The analysis may not capture the full meaning of the messages as they are usually short and informal.")
        st.text("*People often use emojis, slangs, or informal expressions in WhatsApp chats,")
        st.text("which can be difficult for sentiment analysis models to interpret correctly.")
        st.text("*Your WhatsApp messages are encrypted.")
        
                 
                 
            
        
        
            
        
      

            
        