#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


raw_data = pd.read_csv('C:\\Users\\DELL\\Videos\\virus data set\\netflix_titles.csv')


# In[3]:


raw_data


# In[4]:


raw_data.info()


# In[5]:


df=raw_data.copy()


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(['director'],inplace=True,axis=1)
df.drop(['cast'],inplace=True,axis=1)


# In[9]:


df.isnull().sum()


# In[10]:


df['country']=df['country'].fillna('A')


# In[11]:


df.isnull().sum()


# In[12]:


df


# In[13]:


df['type'].unique()


# In[14]:


len(df['type'].unique())


# In[15]:


percent=df['type'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(percent.values,labels=percent.index,autopct='%.1f%%',startangle=90,explode=[0.1,0],colors=['red','purple'])
plt.legend()
plt.title('movie or tv show?',fontsize=14)
plt.show()


# In[16]:


type(df['date_added'])


# In[17]:


type(df['date_added'][0])


# In[18]:


df['date_added']=pd.to_datetime(df['date_added'],format='mixed')


# In[19]:


df['date_added'][0].month


# In[20]:


list_months=[]


# In[21]:


list_months


# In[22]:


for i in range(8807):
    list_months.append(df['date_added'][i].month)


# In[23]:


list_months


# In[24]:


len (list_months)


# In[25]:


df['Month_added']=list_months


# In[26]:


df['date_added'][0].day


# In[27]:


list_day=[]


# In[28]:


list_day


# In[29]:


for i in range(8807):
    list_day.append(df['date_added'][i].day)


# In[30]:


list_day


# In[31]:


df['day_added']=list_day


# In[32]:


df


# In[ ]:





# In[33]:


type_and_date=df[['type','date_added']].copy()


# In[34]:


type_and_date['date_added']=pd.to_datetime(type_and_date['date_added'],format='mixed')


# In[35]:


type_and_date['year_added']=pd.DatetimeIndex(type_and_date['date_added']).year


# In[36]:


type_and_date['month_added']=pd.DatetimeIndex(type_and_date['date_added']).month


# In[ ]:





# In[37]:


display(type_and_date)


# In[ ]:





# In[38]:


type_and_date.isnull().sum()


# In[39]:


type_and_date.dropna(inplace=True)


# In[40]:


type_and_date['year_added'].unique()


# In[41]:


type_and_date['month_added'].unique()


# In[42]:


count=type_and_date.groupby(['year_added','month_added','type']).count().rename(columns={'date_added':'count'})
count.reset_index(inplace=True)
count


# In[43]:


count_Tvshow=count[count['type']=='TV Show']
count_Movie=count[count['type']=='Movie']


# In[44]:


count_Tvshow


# In[45]:


count_Movie


# In[46]:


plt.figure(figsize=(16,10))
plt.plot(count_Movie['year_added'],count_Movie['count'],lw=4,c='red',label='Movies')
plt.plot(count_Tvshow['year_added'],count_Tvshow['count'],lw=4,c='black',label='Tv Show')
plt.legend()
plt.grid()
plt.title('count of movie and tv shows by years', fontsize=18)
plt.show()


# In[47]:


rouded=count_Movie['count'].round().astype(int)
rouded=count_Tvshow['count'].round().astype(int)
plt.figure(figsize=(16,10))
plt.hist(count_Movie['month_added'],bins=10,color='red',label='Movies')
plt.hist(count_Tvshow['month_added'],bins=10,color='blue',label='Tv Show')
plt.xlabel('month',fontsize=14)
plt.ylabel('count',fontsize=14)
plt.legend()
plt.grid()
plt.title('count of movie and tv shows by years', fontsize=18)
plt.show()


# In[48]:


count_Tvshow


# In[49]:


cumulative_count_tvshow = count_Tvshow.groupby('type').cumsum()
cumulative_count_tvshow


# In[50]:


count_Movie


# In[51]:


cumulative_count_movie = count_Movie.groupby('type').cumsum()
cumulative_count_movie


# In[52]:


plt.figure(figsize=(16,10))
plt.plot(count_Movie['year_added'], cumulative_count_movie['count'], lw=4, c='red', label='Movie')
plt.plot(count_Tvshow['year_added'], cumulative_count_tvshow['count'], lw=4, c='purple', label='TV Show')
plt.legend()
plt.grid()
plt.title('Cumulative Count of Movie and TV Show By Year')
plt.show()


# In[53]:


plt.figure(figsize=(16,10))
plt.title('Number of Movies and TV Shows by Rating', fontsize=18)
plt.grid()
sns.countplot(x='rating', data=df, saturation=1, order=df['rating'].value_counts().index);


# In[54]:


plt.figure(figsize=(16,10))
plt.title('Count Rating by TV Show and Movie', fontsize=18)
plt.grid()
sns.countplot(x='rating', hue='type', data=df, order=df['rating'].value_counts().index, palette=['blue', 'gold']);


# In[55]:


df.sort_values('release_year', ascending=True)[['title', 'date_added', 'release_year']].head(10)


# In[56]:


df.sort_values('release_year', ascending=False)[['title', 'date_added', 'release_year']].head(10)


# In[57]:


df['year_added'] = pd.DatetimeIndex(df['date_added']).year


# In[58]:


df.sort_values('year_added', ascending=True)[['title', 'date_added', 'release_year']].head(10)


# In[59]:


df.sort_values('year_added', ascending=False)[['title', 'date_added', 'release_year']].head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:














# In[60]:


df


# In[61]:


dy=df['country'].copy()


# In[62]:


country_count=dy.value_counts().reset_index()


# In[63]:


country_count.columns =['country','count']


# In[64]:


country_count


# In[65]:


where = pd.DataFrame(country_count).sort_values('count', ascending=False)
where


# In[66]:


plt.figure(figsize=(16,10))
plt.title('Number of Production by Countries', fontsize=20)
sns.barplot(x=where['country'].head(15), y=where['count'].head(10), palette='Spectral');


# In[67]:


plt.figure(figsize=(15,11))
plt.title('Top 10 Genres of Movies', fontsize=18)
df[df['type']=="Movie"]['listed_in'].value_counts()[:10].plot(kind='barh', color='gold');


# In[ ]:





# In[ ]:





# In[68]:


plt.figure(figsize=(16,10))
plt.title('Top 10 Genres of Tvshow', fontsize=20)
df[df['type']=="TV Show"]['listed_in'].value_counts()[:10].plot(kind='barh', color='blue');


# In[69]:


from wordcloud import WordCloud


# In[70]:


dx=df['listed_in']


# In[71]:


listed=dx.value_counts().reset_index()


# In[72]:


listed.columns =['listed_in','count']


# In[73]:


listed


# In[ ]:





# In[74]:


dm = df.copy()


# In[75]:


dm


# In[76]:


dm.info()


# In[77]:


dm.drop(['description'],inplace=True,axis=1)
dm.drop(['show_id'],inplace=True,axis=1)
dm.drop(['date_added'],inplace=True,axis=1)
dm.drop(['title'],inplace=True,axis=1)
dm.drop(['duration'],inplace=True,axis=1)


# In[78]:


dm


# In[79]:


dm['type'].unique()


# In[80]:


type_column = pd.get_dummies(dm['type'])


# In[81]:


type_column


# In[82]:


type_column['check']=type_column.sum(axis=1)
type_column


# In[83]:


type_column = pd.get_dummies(dm['type'],drop_first=True)
type_column


# In[84]:


dm.columns.values


# In[85]:


dm=dm.drop(['type'],axis=1)


# In[86]:


dm=pd.concat([dm,type_column],axis=1)


# In[87]:


dm


# In[88]:


dm['listed_in'].unique()


# In[89]:


sorted(dm['listed_in'].unique())


# In[90]:


sorted(dm['country'].unique())


# In[91]:


rate_column = pd.get_dummies(dm['rating'])


# In[92]:


rate_column


# In[93]:


rate_column['check']=rate_column.sum(axis=1)
rate_column


# In[94]:


rate_column = pd.get_dummies(dm['rating'],drop_first=True)
rate_column


# In[95]:


dm.columns.values


# In[96]:


dm=dm.drop(['rating'],axis=1)


# In[97]:


dm=pd.concat([dm,rate_column],axis=1)


# In[98]:


dm


# In[ ]:





# In[99]:


top_country=dm['country'].value_counts().head(15).index.tolist()
for category in top_country:
    dm[category+'_dummy']=(dm['country']==category).astype(bool)


# In[100]:


dm


# In[ ]:





# In[101]:


list_column = pd.get_dummies(dm['listed_in'],drop_first=True)


# In[102]:


list_column


# In[103]:


list_column['check']=rate_column.sum(axis=1)
list_column


# In[104]:


list_column['check'].sum(axis=0)


# In[105]:


list_column['check'].unique()


# In[106]:


dm=pd.concat([dm,list_column],axis=1)


# In[107]:


dm


# In[108]:


dm['TV Show'].count()


# In[109]:


dm.columns.values


# In[110]:


dm.drop(['country'],inplace=True,axis=1)
dm.drop(['listed_in'],inplace=True,axis=1)
dm.drop(['check'],inplace=True,axis=1)


# In[111]:


dm


# In[112]:


ass= dm.copy()


# In[113]:


ass


# In[114]:


random_row=ass.sample(n=1000)


# In[115]:


random_row


# In[116]:


random_row.columns.values


# In[117]:


random_row.info()


# In[118]:


missing_values=random_row.isnull().sum()
print("missing value:\n",missing_values)


# In[119]:


random_row=random_row.fillna(0)


# In[120]:


random_row.isnull().sum()


# In[ ]:





# In[ ]:





# In[121]:


ass


# In[122]:


missing_values=ass.isnull().sum()
print("missing value:\n",missing_values)


# In[123]:


ass=ass.fillna(0)


# In[124]:


ass.isnull().sum()


# In[ ]:





# In[125]:


dg =ass.head(1000)


# In[ ]:





# In[126]:


dp=ass.iloc[1000:2000]


# In[127]:


dp


# In[128]:


dp.columns.values


# In[ ]:





# In[ ]:





# In[ ]:





# In[133]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
X = random_row[['release_year', 'Month_added', 'day_added', 'year_added',
        '74 min', '84 min', 'G', 'NC-17', 'NR', 'PG', 'PG-13',
       'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y', 'TV-Y7',
       'TV-Y7-FV', 'UR', 'United States_dummy', 'India_dummy', 'A_dummy',
       'United Kingdom_dummy', 'Japan_dummy', 'South Korea_dummy',
       'Canada_dummy', 'Spain_dummy', 'France_dummy', 'Mexico_dummy',
       'Egypt_dummy', 'Turkey_dummy', 'Nigeria_dummy', 'Australia_dummy',
       'Taiwan_dummy', 'Action & Adventure, Anime Features',
       'Action & Adventure, Anime Features, Children & Family Movies',
       'Action & Adventure, Anime Features, Classic Movies',
       'Action & Adventure, Anime Features, Horror Movies',
       'Action & Adventure, Anime Features, International Movies',
       'Action & Adventure, Anime Features, Sci-Fi & Fantasy',
       'Action & Adventure, Children & Family Movies',
       'Action & Adventure, Children & Family Movies, Classic Movies',
       'Action & Adventure, Children & Family Movies, Comedies',
       'Action & Adventure, Children & Family Movies, Cult Movies',
       'Action & Adventure, Children & Family Movies, Dramas',
       'Action & Adventure, Children & Family Movies, Independent Movies',
       'Action & Adventure, Children & Family Movies, Sci-Fi & Fantasy',
       'Action & Adventure, Classic Movies',
       'Action & Adventure, Classic Movies, Comedies',
       'Action & Adventure, Classic Movies, Cult Movies',
       'Action & Adventure, Classic Movies, Dramas',
       'Action & Adventure, Classic Movies, International Movies',
       'Action & Adventure, Classic Movies, Sci-Fi & Fantasy',
       'Action & Adventure, Comedies',
       'Action & Adventure, Comedies, Cult Movies',
       'Action & Adventure, Comedies, Dramas',
       'Action & Adventure, Comedies, Horror Movies',
       'Action & Adventure, Comedies, Independent Movies',
       'Action & Adventure, Comedies, International Movies',
       'Action & Adventure, Comedies, Music & Musicals',
       'Action & Adventure, Comedies, Romantic Movies',
       'Action & Adventure, Comedies, Sci-Fi & Fantasy',
       'Action & Adventure, Comedies, Sports Movies',
       'Action & Adventure, Cult Movies',
       'Action & Adventure, Cult Movies, Dramas',
       'Action & Adventure, Cult Movies, International Movies',
       'Action & Adventure, Cult Movies, Sci-Fi & Fantasy',
       'Action & Adventure, Documentaries, International Movies',
       'Action & Adventure, Documentaries, Sports Movies',
       'Action & Adventure, Dramas',
       'Action & Adventure, Dramas, Faith & Spirituality',
       'Action & Adventure, Dramas, Independent Movies',
       'Action & Adventure, Dramas, International Movies',
       'Action & Adventure, Dramas, Romantic Movies',
       'Action & Adventure, Dramas, Sci-Fi & Fantasy',
       'Action & Adventure, Dramas, Sports Movies',
       'Action & Adventure, Faith & Spirituality, Sci-Fi & Fantasy',
       'Action & Adventure, Horror Movies',
       'Action & Adventure, Horror Movies, Independent Movies',
       'Action & Adventure, Horror Movies, International Movies',
       'Action & Adventure, Horror Movies, Sci-Fi & Fantasy',
       'Action & Adventure, Horror Movies, Thrillers',
       'Action & Adventure, Independent Movies',
       'Action & Adventure, Independent Movies, International Movies',
       'Action & Adventure, Independent Movies, Sci-Fi & Fantasy',
       'Action & Adventure, International Movies',
       'Action & Adventure, International Movies, Music & Musicals',
       'Action & Adventure, International Movies, Romantic Movies',
       'Action & Adventure, International Movies, Sci-Fi & Fantasy',
       'Action & Adventure, International Movies, Sports Movies',
       'Action & Adventure, International Movies, Thrillers',
       'Action & Adventure, Romantic Movies',
       'Action & Adventure, Romantic Movies, Sci-Fi & Fantasy',
       'Action & Adventure, Sci-Fi & Fantasy',
       'Action & Adventure, Sci-Fi & Fantasy, Sports Movies',
       'Action & Adventure, Sci-Fi & Fantasy, Thrillers',
       'Action & Adventure, Sports Movies',
       'Action & Adventure, Thrillers', 'Anime Features',
       'Anime Features, Children & Family Movies',
       'Anime Features, Children & Family Movies, International Movies',
       'Anime Features, Documentaries',
       'Anime Features, International Movies',
       'Anime Features, International Movies, Romantic Movies',
       'Anime Features, International Movies, Sci-Fi & Fantasy',
       'Anime Features, Music & Musicals',
       'Anime Features, Music & Musicals, Sci-Fi & Fantasy',
       'Anime Features, Romantic Movies', 'Anime Series',
       'Anime Series, Crime TV Shows',
       'Anime Series, Crime TV Shows, International TV Shows',
       'Anime Series, Crime TV Shows, TV Horror',
       'Anime Series, Crime TV Shows, TV Thrillers',
       'Anime Series, International TV Shows',
       'Anime Series, International TV Shows, Romantic TV Shows',
       'Anime Series, International TV Shows, Spanish-Language TV Shows',
       'Anime Series, International TV Shows, TV Horror',
       'Anime Series, International TV Shows, TV Thrillers',
       'Anime Series, International TV Shows, Teen TV Shows',
       "Anime Series, Kids' TV",
       "Anime Series, Kids' TV, TV Action & Adventure",
       'Anime Series, Romantic TV Shows',
       'Anime Series, Romantic TV Shows, Teen TV Shows',
       'Anime Series, Stand-Up Comedy & Talk Shows',
       'Anime Series, TV Horror, TV Thrillers',
       'Anime Series, Teen TV Shows',
       'British TV Shows, Classic & Cult TV, International TV Shows',
       "British TV Shows, Classic & Cult TV, Kids' TV",
       'British TV Shows, Classic & Cult TV, TV Comedies',
       'British TV Shows, Crime TV Shows, Docuseries',
       'British TV Shows, Crime TV Shows, International TV Shows',
       'British TV Shows, Crime TV Shows, TV Dramas',
       'British TV Shows, Docuseries',
       'British TV Shows, Docuseries, International TV Shows',
       'British TV Shows, Docuseries, Reality TV',
       'British TV Shows, Docuseries, Science & Nature TV',
       'British TV Shows, Docuseries, TV Comedies',
       'British TV Shows, International TV Shows, Reality TV',
       'British TV Shows, International TV Shows, Romantic TV Shows',
       'British TV Shows, International TV Shows, Stand-Up Comedy & Talk Shows',
       'British TV Shows, International TV Shows, TV Action & Adventure',
       'British TV Shows, International TV Shows, TV Comedies',
       'British TV Shows, International TV Shows, TV Dramas',
       "British TV Shows, Kids' TV",
       "British TV Shows, Kids' TV, TV Comedies",
       "British TV Shows, Kids' TV, TV Dramas",
       "British TV Shows, Kids' TV, TV Thrillers",
       'British TV Shows, Reality TV',
       'British TV Shows, Reality TV, Romantic TV Shows',
       'British TV Shows, Romantic TV Shows, TV Dramas',
       'British TV Shows, TV Comedies',
       'British TV Shows, TV Comedies, TV Dramas',
       'British TV Shows, TV Dramas, TV Sci-Fi & Fantasy',
       'British TV Shows, TV Horror, TV Thrillers',
       'Children & Family Movies',
       'Children & Family Movies, Classic Movies',
       'Children & Family Movies, Classic Movies, Comedies',
       'Children & Family Movies, Classic Movies, Dramas',
       'Children & Family Movies, Comedies',
       'Children & Family Movies, Comedies, Cult Movies',
       'Children & Family Movies, Comedies, Dramas',
       'Children & Family Movies, Comedies, Faith & Spirituality',
       'Children & Family Movies, Comedies, International Movies',
       'Children & Family Movies, Comedies, LGBTQ Movies',
       'Children & Family Movies, Comedies, Music & Musicals',
       'Children & Family Movies, Comedies, Romantic Movies',
       'Children & Family Movies, Comedies, Sci-Fi & Fantasy',
       'Children & Family Movies, Comedies, Sports Movies',
       'Children & Family Movies, Documentaries',
       'Children & Family Movies, Documentaries, International Movies',
       'Children & Family Movies, Documentaries, Sports Movies',
       'Children & Family Movies, Dramas',
       'Children & Family Movies, Dramas, Faith & Spirituality',
       'Children & Family Movies, Dramas, Independent Movies',
       'Children & Family Movies, Dramas, International Movies',
       'Children & Family Movies, Dramas, Music & Musicals',
       'Children & Family Movies, Dramas, Romantic Movies',
       'Children & Family Movies, Dramas, Sports Movies',
       'Children & Family Movies, Faith & Spirituality',
       'Children & Family Movies, Faith & Spirituality, Music & Musicals',
       'Children & Family Movies, Independent Movies',
       'Children & Family Movies, Music & Musicals',
       'Children & Family Movies, Sci-Fi & Fantasy',
       'Children & Family Movies, Sports Movies',
       'Classic & Cult TV, Crime TV Shows, International TV Shows',
       'Classic & Cult TV, Crime TV Shows, TV Dramas',
       "Classic & Cult TV, Kids' TV, Spanish-Language TV Shows",
       "Classic & Cult TV, Kids' TV, TV Action & Adventure",
       "Classic & Cult TV, Kids' TV, TV Comedies",
       'Classic & Cult TV, TV Action & Adventure, TV Dramas',
       'Classic & Cult TV, TV Action & Adventure, TV Horror',
       'Classic & Cult TV, TV Action & Adventure, TV Sci-Fi & Fantasy',
       'Classic & Cult TV, TV Comedies',
       'Classic & Cult TV, TV Dramas, TV Sci-Fi & Fantasy',
       'Classic & Cult TV, TV Horror, TV Mysteries',
       'Classic & Cult TV, TV Sci-Fi & Fantasy',
       'Classic Movies, Comedies, Cult Movies',
       'Classic Movies, Comedies, Dramas',
       'Classic Movies, Comedies, Independent Movies',
       'Classic Movies, Comedies, International Movies',
       'Classic Movies, Comedies, Music & Musicals',
       'Classic Movies, Comedies, Romantic Movies',
       'Classic Movies, Comedies, Sports Movies',
       'Classic Movies, Cult Movies, Documentaries',
       'Classic Movies, Cult Movies, Dramas',
       'Classic Movies, Cult Movies, Horror Movies',
       'Classic Movies, Documentaries', 'Classic Movies, Dramas',
       'Classic Movies, Dramas, Independent Movies',
       'Classic Movies, Dramas, International Movies',
       'Classic Movies, Dramas, LGBTQ Movies',
       'Classic Movies, Dramas, Music & Musicals',
       'Classic Movies, Dramas, Romantic Movies',
       'Classic Movies, Dramas, Sports Movies',
       'Classic Movies, Horror Movies, Thrillers',
       'Classic Movies, Independent Movies, Thrillers',
       'Classic Movies, Music & Musicals', 'Classic Movies, Thrillers',
       'Comedies', 'Comedies, Cult Movies',
       'Comedies, Cult Movies, Dramas',
       'Comedies, Cult Movies, Horror Movies',
       'Comedies, Cult Movies, Independent Movies',
       'Comedies, Cult Movies, International Movies',
       'Comedies, Cult Movies, LGBTQ Movies',
       'Comedies, Cult Movies, Music & Musicals',
       'Comedies, Cult Movies, Sci-Fi & Fantasy',
       'Comedies, Cult Movies, Sports Movies', 'Comedies, Documentaries',
       'Comedies, Documentaries, International Movies',
       'Comedies, Dramas', 'Comedies, Dramas, Faith & Spirituality',
       'Comedies, Dramas, Independent Movies',
       'Comedies, Dramas, International Movies',
       'Comedies, Dramas, LGBTQ Movies',
       'Comedies, Dramas, Music & Musicals',
       'Comedies, Dramas, Romantic Movies',
       'Comedies, Dramas, Sports Movies',
       'Comedies, Faith & Spirituality, International Movies',
       'Comedies, Faith & Spirituality, Romantic Movies',
       'Comedies, Horror Movies',
       'Comedies, Horror Movies, Independent Movies',
       'Comedies, Horror Movies, International Movies',
       'Comedies, Horror Movies, Sci-Fi & Fantasy',
       'Comedies, Independent Movies',
       'Comedies, Independent Movies, International Movies',
       'Comedies, Independent Movies, LGBTQ Movies',
       'Comedies, Independent Movies, Music & Musicals',
       'Comedies, Independent Movies, Romantic Movies',
       'Comedies, Independent Movies, Thrillers',
       'Comedies, International Movies',
       'Comedies, International Movies, LGBTQ Movies',
       'Comedies, International Movies, Music & Musicals',
       'Comedies, International Movies, Romantic Movies',
       'Comedies, International Movies, Sci-Fi & Fantasy',
       'Comedies, International Movies, Sports Movies',
       'Comedies, International Movies, Thrillers',
       'Comedies, LGBTQ Movies',
       'Comedies, LGBTQ Movies, Music & Musicals',
       'Comedies, LGBTQ Movies, Thrillers', 'Comedies, Music & Musicals',
       'Comedies, Music & Musicals, Romantic Movies',
       'Comedies, Music & Musicals, Sports Movies',
       'Comedies, Romantic Movies',
       'Comedies, Romantic Movies, Sports Movies',
       'Comedies, Sci-Fi & Fantasy', 'Comedies, Sports Movies',
       'Crime TV Shows, Docuseries',
       'Crime TV Shows, Docuseries, International TV Shows',
       'Crime TV Shows, Docuseries, Science & Nature TV',
       'Crime TV Shows, Docuseries, TV Mysteries',
       'Crime TV Shows, International TV Shows, Korean TV Shows',
       'Crime TV Shows, International TV Shows, Reality TV',
       'Crime TV Shows, International TV Shows, Romantic TV Shows',
       'Crime TV Shows, International TV Shows, Spanish-Language TV Shows',
       'Crime TV Shows, International TV Shows, TV Action & Adventure',
       'Crime TV Shows, International TV Shows, TV Comedies',
       'Crime TV Shows, International TV Shows, TV Dramas',
       'Crime TV Shows, International TV Shows, TV Mysteries',
       'Crime TV Shows, International TV Shows, TV Sci-Fi & Fantasy',
       'Crime TV Shows, International TV Shows, TV Thrillers',
       "Crime TV Shows, Kids' TV",
       "Crime TV Shows, Kids' TV, TV Comedies",
       'Crime TV Shows, Romantic TV Shows, Spanish-Language TV Shows',
       'Crime TV Shows, Romantic TV Shows, TV Dramas',
       'Crime TV Shows, Spanish-Language TV Shows, TV Action & Adventure',
       'Crime TV Shows, Spanish-Language TV Shows, TV Dramas',
       'Crime TV Shows, TV Action & Adventure',
       'Crime TV Shows, TV Action & Adventure, TV Comedies',
       'Crime TV Shows, TV Action & Adventure, TV Dramas',
       'Crime TV Shows, TV Action & Adventure, TV Sci-Fi & Fantasy',
       'Crime TV Shows, TV Action & Adventure, TV Thrillers',
       'Crime TV Shows, TV Comedies',
       'Crime TV Shows, TV Comedies, TV Dramas',
       'Crime TV Shows, TV Comedies, Teen TV Shows',
       'Crime TV Shows, TV Dramas',
       'Crime TV Shows, TV Dramas, TV Horror',
       'Crime TV Shows, TV Dramas, TV Mysteries',
       'Crime TV Shows, TV Dramas, TV Thrillers',
       'Crime TV Shows, TV Horror, TV Mysteries',
       'Cult Movies, Dramas, International Movies',
       'Cult Movies, Dramas, Music & Musicals',
       'Cult Movies, Dramas, Thrillers', 'Cult Movies, Horror Movies',
       'Cult Movies, Horror Movies, Independent Movies',
       'Cult Movies, Horror Movies, Thrillers',
       'Cult Movies, Independent Movies, Thrillers', 'Documentaries',
       'Documentaries, Dramas',
       'Documentaries, Dramas, International Movies',
       'Documentaries, Faith & Spirituality',
       'Documentaries, Faith & Spirituality, International Movies',
       'Documentaries, Faith & Spirituality, Music & Musicals',
       'Documentaries, Horror Movies',
       'Documentaries, International Movies',
       'Documentaries, International Movies, LGBTQ Movies',
       'Documentaries, International Movies, Music & Musicals',
       'Documentaries, International Movies, Sports Movies',
       'Documentaries, LGBTQ Movies',
       'Documentaries, LGBTQ Movies, Music & Musicals',
       'Documentaries, LGBTQ Movies, Sports Movies',
       'Documentaries, Music & Musicals', 'Documentaries, Sports Movies',
       'Documentaries, Stand-Up Comedy', 'Docuseries',
       'Docuseries, International TV Shows',
       'Docuseries, International TV Shows, Reality TV',
       'Docuseries, International TV Shows, Science & Nature TV',
       'Docuseries, International TV Shows, Spanish-Language TV Shows',
       "Docuseries, Kids' TV, Science & Nature TV",
       'Docuseries, Reality TV',
       'Docuseries, Reality TV, Science & Nature TV',
       'Docuseries, Reality TV, Teen TV Shows',
       'Docuseries, Science & Nature TV',
       'Docuseries, Science & Nature TV, TV Action & Adventure',
       'Docuseries, Science & Nature TV, TV Comedies',
       'Docuseries, Science & Nature TV, TV Dramas',
       'Docuseries, Spanish-Language TV Shows',
       'Docuseries, Stand-Up Comedy & Talk Shows',
       'Docuseries, TV Comedies', 'Docuseries, TV Dramas',
       'Docuseries, TV Sci-Fi & Fantasy', 'Dramas',
       'Dramas, Faith & Spirituality',
       'Dramas, Faith & Spirituality, Independent Movies',
       'Dramas, Faith & Spirituality, International Movies',
       'Dramas, Faith & Spirituality, Romantic Movies',
       'Dramas, Faith & Spirituality, Sports Movies',
       'Dramas, Horror Movies, Music & Musicals',
       'Dramas, Horror Movies, Sci-Fi & Fantasy',
       'Dramas, Horror Movies, Thrillers', 'Dramas, Independent Movies',
       'Dramas, Independent Movies, International Movies',
       'Dramas, Independent Movies, LGBTQ Movies',
       'Dramas, Independent Movies, Music & Musicals',
       'Dramas, Independent Movies, Romantic Movies',
       'Dramas, Independent Movies, Sci-Fi & Fantasy',
       'Dramas, Independent Movies, Sports Movies',
       'Dramas, Independent Movies, Thrillers',
       'Dramas, International Movies',
       'Dramas, International Movies, LGBTQ Movies',
       'Dramas, International Movies, Music & Musicals',
       'Dramas, International Movies, Romantic Movies',
       'Dramas, International Movies, Sci-Fi & Fantasy',
       'Dramas, International Movies, Sports Movies',
       'Dramas, International Movies, Thrillers', 'Dramas, LGBTQ Movies',
       'Dramas, LGBTQ Movies, Romantic Movies',
       'Dramas, Music & Musicals',
       'Dramas, Music & Musicals, Romantic Movies',
       'Dramas, Romantic Movies',
       'Dramas, Romantic Movies, Sci-Fi & Fantasy',
       'Dramas, Romantic Movies, Sports Movies',
       'Dramas, Romantic Movies, Thrillers', 'Dramas, Sci-Fi & Fantasy',
       'Dramas, Sci-Fi & Fantasy, Thrillers', 'Dramas, Sports Movies',
       'Dramas, Thrillers', 'Horror Movies',
       'Horror Movies, Independent Movies',
       'Horror Movies, Independent Movies, International Movies',
       'Horror Movies, Independent Movies, Sci-Fi & Fantasy',
       'Horror Movies, Independent Movies, Thrillers',
       'Horror Movies, International Movies',
       'Horror Movies, International Movies, Romantic Movies',
       'Horror Movies, International Movies, Sci-Fi & Fantasy',
       'Horror Movies, International Movies, Thrillers',
       'Horror Movies, LGBTQ Movies',
       'Horror Movies, LGBTQ Movies, Music & Musicals',
       'Horror Movies, Romantic Movies, Sci-Fi & Fantasy',
       'Horror Movies, Sci-Fi & Fantasy',
       'Horror Movies, Sci-Fi & Fantasy, Thrillers',
       'Horror Movies, Thrillers', 'Independent Movies',
       'Independent Movies, International Movies, Thrillers',
       'Independent Movies, Sci-Fi & Fantasy, Thrillers',
       'Independent Movies, Thrillers', 'International Movies',
       'International Movies, LGBTQ Movies, Romantic Movies',
       'International Movies, Music & Musicals',
       'International Movies, Music & Musicals, Romantic Movies',
       'International Movies, Music & Musicals, Thrillers',
       'International Movies, Romantic Movies',
       'International Movies, Romantic Movies, Sci-Fi & Fantasy',
       'International Movies, Romantic Movies, Thrillers',
       'International Movies, Sci-Fi & Fantasy',
       'International Movies, Sci-Fi & Fantasy, Thrillers',
       'International Movies, Sports Movies',
       'International Movies, Thrillers', 'International TV Shows',
       "International TV Shows, Kids' TV, TV Mysteries",
       'International TV Shows, Korean TV Shows, Reality TV',
       'International TV Shows, Korean TV Shows, Romantic TV Shows',
       'International TV Shows, Korean TV Shows, Stand-Up Comedy & Talk Shows',
       'International TV Shows, Korean TV Shows, TV Action & Adventure',
       'International TV Shows, Korean TV Shows, TV Comedies',
       'International TV Shows, Korean TV Shows, TV Dramas',
       'International TV Shows, Korean TV Shows, TV Horror',
       'International TV Shows, Reality TV',
       'International TV Shows, Reality TV, Romantic TV Shows',
       'International TV Shows, Reality TV, Spanish-Language TV Shows',
       'International TV Shows, Reality TV, TV Action & Adventure',
       'International TV Shows, Reality TV, TV Comedies',
       'International TV Shows, Romantic TV Shows',
       'International TV Shows, Romantic TV Shows, Spanish-Language TV Shows',
       'International TV Shows, Romantic TV Shows, TV Action & Adventure',
       'International TV Shows, Romantic TV Shows, TV Comedies',
       'International TV Shows, Romantic TV Shows, TV Dramas',
       'International TV Shows, Romantic TV Shows, TV Mysteries',
       'International TV Shows, Romantic TV Shows, Teen TV Shows',
       'International TV Shows, Spanish-Language TV Shows, Stand-Up Comedy & Talk Shows',
       'International TV Shows, Spanish-Language TV Shows, TV Action & Adventure',
       'International TV Shows, Spanish-Language TV Shows, TV Comedies',
       'International TV Shows, Spanish-Language TV Shows, TV Dramas',
       'International TV Shows, Spanish-Language TV Shows, TV Horror',
       'International TV Shows, Stand-Up Comedy & Talk Shows',
       'International TV Shows, Stand-Up Comedy & Talk Shows, TV Comedies',
       'International TV Shows, TV Action & Adventure, TV Comedies',
       'International TV Shows, TV Action & Adventure, TV Dramas',
       'International TV Shows, TV Action & Adventure, TV Horror',
       'International TV Shows, TV Action & Adventure, TV Mysteries',
       'International TV Shows, TV Action & Adventure, TV Sci-Fi & Fantasy',
       'International TV Shows, TV Comedies',
       'International TV Shows, TV Comedies, TV Dramas',
       'International TV Shows, TV Comedies, TV Sci-Fi & Fantasy',
       'International TV Shows, TV Dramas',
       'International TV Shows, TV Dramas, TV Horror',
       'International TV Shows, TV Dramas, TV Mysteries',
       'International TV Shows, TV Dramas, TV Sci-Fi & Fantasy',
       'International TV Shows, TV Dramas, TV Thrillers',
       'International TV Shows, TV Dramas, Teen TV Shows',
       'International TV Shows, TV Horror, TV Mysteries',
       'International TV Shows, TV Horror, TV Sci-Fi & Fantasy',
       'International TV Shows, TV Horror, TV Thrillers',
       'International TV Shows, TV Mysteries, TV Thrillers', "Kids' TV",
       "Kids' TV, Korean TV Shows",
       "Kids' TV, Korean TV Shows, TV Comedies",
       "Kids' TV, Reality TV, Science & Nature TV",
       "Kids' TV, Reality TV, TV Dramas",
       "Kids' TV, Spanish-Language TV Shows",
       "Kids' TV, Spanish-Language TV Shows, Teen TV Shows",
       "Kids' TV, TV Action & Adventure",
       "Kids' TV, TV Action & Adventure, TV Comedies",
       "Kids' TV, TV Action & Adventure, TV Dramas",
       "Kids' TV, TV Action & Adventure, TV Sci-Fi & Fantasy",
       "Kids' TV, TV Comedies", "Kids' TV, TV Comedies, TV Dramas",
       "Kids' TV, TV Comedies, TV Sci-Fi & Fantasy",
       "Kids' TV, TV Comedies, Teen TV Shows", "Kids' TV, TV Dramas",
       "Kids' TV, TV Dramas, Teen TV Shows",
       "Kids' TV, TV Sci-Fi & Fantasy", "Kids' TV, TV Thrillers",
       'LGBTQ Movies, Thrillers', 'Movies', 'Music & Musicals',
       'Music & Musicals, Romantic Movies',
       'Music & Musicals, Stand-Up Comedy', 'Reality TV',
       'Reality TV, Romantic TV Shows', 'Reality TV, Science & Nature TV',
       'Reality TV, Science & Nature TV, TV Action & Adventure',
       'Reality TV, Spanish-Language TV Shows',
       'Reality TV, TV Action & Adventure, TV Mysteries',
       'Reality TV, TV Comedies', 'Reality TV, TV Comedies, TV Horror',
       'Reality TV, TV Horror, TV Thrillers', 'Reality TV, Teen TV Shows',
       'Romantic Movies',
       'Romantic TV Shows, Spanish-Language TV Shows, TV Comedies',
       'Romantic TV Shows, Spanish-Language TV Shows, TV Dramas',
       'Romantic TV Shows, TV Action & Adventure, TV Dramas',
       'Romantic TV Shows, TV Comedies',
       'Romantic TV Shows, TV Comedies, TV Dramas',
       'Romantic TV Shows, TV Dramas',
       'Romantic TV Shows, TV Dramas, TV Sci-Fi & Fantasy',
       'Romantic TV Shows, TV Dramas, Teen TV Shows',
       'Romantic TV Shows, Teen TV Shows', 'Sci-Fi & Fantasy',
       'Sci-Fi & Fantasy, Thrillers',
       'Spanish-Language TV Shows, TV Dramas', 'Sports Movies',
       'Stand-Up Comedy', 'Stand-Up Comedy & Talk Shows',
       'Stand-Up Comedy & Talk Shows, TV Comedies',
       'Stand-Up Comedy & Talk Shows, TV Mysteries, TV Sci-Fi & Fantasy',
       'TV Action & Adventure', 'TV Action & Adventure, TV Comedies',
       'TV Action & Adventure, TV Comedies, TV Dramas',
       'TV Action & Adventure, TV Comedies, TV Horror',
       'TV Action & Adventure, TV Comedies, TV Sci-Fi & Fantasy',
       'TV Action & Adventure, TV Dramas',
       'TV Action & Adventure, TV Dramas, TV Horror',
       'TV Action & Adventure, TV Dramas, TV Mysteries',
       'TV Action & Adventure, TV Dramas, TV Sci-Fi & Fantasy',
       'TV Action & Adventure, TV Dramas, Teen TV Shows',
       'TV Action & Adventure, TV Horror, TV Sci-Fi & Fantasy',
       'TV Action & Adventure, TV Mysteries, TV Sci-Fi & Fantasy',
       'TV Action & Adventure, TV Sci-Fi & Fantasy', 'TV Comedies',
       'TV Comedies, TV Dramas', 'TV Comedies, TV Dramas, TV Horror',
       'TV Comedies, TV Dramas, TV Mysteries',
       'TV Comedies, TV Dramas, TV Sci-Fi & Fantasy',
       'TV Comedies, TV Dramas, Teen TV Shows',
       'TV Comedies, TV Horror, TV Thrillers',
       'TV Comedies, TV Mysteries', 'TV Comedies, TV Sci-Fi & Fantasy',
       'TV Comedies, TV Sci-Fi & Fantasy, Teen TV Shows',
       'TV Comedies, Teen TV Shows', 'TV Dramas',
       'TV Dramas, TV Horror, TV Mysteries',
       'TV Dramas, TV Mysteries, TV Sci-Fi & Fantasy',
       'TV Dramas, TV Mysteries, TV Thrillers',
       'TV Dramas, TV Sci-Fi & Fantasy',
       'TV Dramas, TV Sci-Fi & Fantasy, TV Thrillers',
       'TV Dramas, TV Sci-Fi & Fantasy, Teen TV Shows',
       'TV Dramas, TV Thrillers', 'TV Dramas, Teen TV Shows',
       'TV Horror, TV Mysteries, TV Sci-Fi & Fantasy',
       'TV Horror, TV Mysteries, TV Thrillers',
       'TV Horror, TV Mysteries, Teen TV Shows',
       'TV Horror, Teen TV Shows', 'TV Sci-Fi & Fantasy, TV Thrillers',
       'TV Shows', 'Thrillers']]
y = random_row['TV Show']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(solver='liblinear')


# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))


# In[134]:


ass


# In[135]:


ass=ass.drop(['TV Show'],axis=1)


# In[136]:


# Assuming 'X_new' is your new data and 'model' is your trained model

# Importing necessary library
from sklearn import metrics

# Make predictions
predictions = model.predict()

# Print the predicted outcomes
print(predictions)


# In[137]:


len(predictions)


# In[ ]:





# In[138]:


import pandas as pd

# Assuming 'data' is your original dataset and 'predictions' is your array of predictions
ass['prediction_column'] = predictions


# In[139]:


ass['prediction_column']


# In[ ]:





# In[ ]:





# In[141]:


percent=ass['prediction_column'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(percent.values,labels=percent.index,autopct='%.1f%%',startangle=90,explode=[0.1,0],colors=['red','purple'])
plt.legend()
plt.title('movie or tv show? prediction',fontsize=14)
plt.show()


# In[142]:


dq=ass['prediction_column'].copy()


# In[143]:


pred=dq.value_counts().reset_index()


# In[144]:


pred.columns =['pediction','count']


# In[145]:


pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[146]:


ass['prediction_column'].count()


# In[147]:


ass


# In[148]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'data' is your dataset with three features and 'predictions' is your array of predictions
# 'data' should have columns for the three features and the prediction results
# For example:
# data = pd.DataFrame({'feature1': feature1_values, 'feature2': feature2_values, 'feature3': feature3_values})

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract features
f1 = ass['Month_added']
f2 = dm['United States_dummy']
f3 = ass['TV Action & Adventure, TV Comedies, TV Dramas']

# Plot the data points
ax.scatter(f1, f2, f3, c=predictions)

# Set labels and title
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F3')
ax.set_title('3D Scatter Plot of Predictions')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




