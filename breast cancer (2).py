#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


raw_txt=('C:\\Users\\DELL\\Downloads\\Breast_Cancer.txt')
with open(raw_txt,'r') as file:
    content=file.read()
    


# In[3]:


df=pd.read_csv(raw_txt,sep=',')


# In[4]:


df


# In[284]:





# In[5]:


df.columns.values


# In[ ]:


#analysis focus.


# In[6]:


dp=df[['Age','Race','Marital Status','differentiate','Survival Months','Status','Tumor Size']]


# In[7]:


dp.head()


# In[8]:


df.describe()


# In[9]:


df.describe(include=object)


# In[10]:


df.mean(numeric_only=True)


# In[11]:


df.std(numeric_only=True)


# In[12]:


df.median(numeric_only=True)


# In[13]:


df.var(numeric_only=True)


# In[14]:


df.mode(numeric_only=True)


# In[15]:


df.skew(numeric_only=True)


# In[16]:


df.shape


# In[17]:


dfA=df[['Marital Status','Status']]


# In[18]:


dfA


# In[19]:


dfB=df[['Age','Status']]


# In[20]:


dfB


# In[21]:


dfC=df[['Race','Tumor Size']]


# In[22]:


dfC


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


plt.figure(figsize=(16,10))
plt.title('AGE', fontsize=18)
plt.grid()
sns.countplot(x='Age', data=df, saturation=1, order=df['Age'].value_counts().index);


# In[25]:


sorted_df=sorted(df['Age'].unique())


# In[26]:


sorted_df


# In[27]:


plt.figure(figsize=(16,10))
plt.title('AGE', fontsize=18)
plt.grid()
sns.countplot(x='Age', data=df, saturation=1, order=sorted_df);


# In[28]:


#outlinner in age since our STD is 9 plus and minus 1 for erro we work with 10 the distance from the mean multiply by two is our outlinner so 20 minu mean and 20 plus mean is our outliner
left_outliner = df['Age'].mean()-20
right_outliner =df['Age'].mean()+20


# In[29]:


print("left_outliner:",left_outliner,"|","right_outliner:",right_outliner)


# In[ ]:





# In[30]:


plt.figure(figsize=(16,10))
plt.title('', fontsize=18)
plt.grid()
sns.countplot(x='Tumor Size', data=df, saturation=1, order=df['Tumor Size'].value_counts().index);


# In[31]:


sorted_dfT=sorted(df['Tumor Size'].unique())


# In[32]:


sorted_dfT


# In[33]:


plt.figure(figsize=(16,10))
plt.title('Tumor', fontsize=18)
plt.grid()
sns.countplot(x='Tumor Size', data=df, saturation=1, order=sorted_dfT);


# In[34]:


left_outliner = df['Tumor Size'].mean()-63
right_outliner =df['Tumor Size'].mean()+63


# In[35]:


print("left_outliner:",left_outliner,"|","right_outliner:",right_outliner)


# In[36]:


#how to drop outliner 
dm=df.copy()


# In[37]:


z_scores = (dm['Age']-dm['Age'].mean())/dm['Age'].std()
threshold=2


# In[38]:


z_scores


# In[39]:


dmA=dm.drop(dm[abs(z_scores)>threshold].index)


# In[40]:


dmA


# In[41]:


z_scoresT = (dm['Tumor Size']-dm['Tumor Size'].mean())/dm['Tumor Size'].std()
threshold=3


# In[42]:


z_scoresT


# In[43]:


dmT=dm.drop(dm[abs(z_scoresT)>threshold].index)


# In[44]:


dmT


# In[45]:


plt.figure(figsize=(10,5))
plt.plot(df['Age'],label='Age')
plt.plot(dmA['Age'],label='concentration Age')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Age Noise')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


#dropping very young people far from average age


# In[47]:


#you can see the filther outliner in the graph above


# In[48]:


plt.figure(figsize=(10,5))
plt.plot(df['Tumor Size'],label='Tumor')
plt.plot(dmT['Tumor Size'],label='Tumor Size')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Tumor Size Noise')
plt.legend()
plt.grid(True)
plt.show()


# In[49]:


#dropping size that are large and far from the average size


# In[50]:


df['Survival Months']


# In[51]:


plt.figure(figsize=(16,10))
plt.title('Survival Months', fontsize=18)
plt.grid()
sns.countplot(x='Survival Months', data=df, saturation=1, order=df['Survival Months'].value_counts().index);


# In[52]:


sorted_dfS=sorted(df['Survival Months'].unique())


# In[53]:


sorted_dfS


# In[54]:


plt.figure(figsize=(16,10))
plt.title('Survival Months', fontsize=18)
plt.grid()
sns.countplot(x='Survival Months', data=df, saturation=1, order=sorted_dfS);


# In[55]:


left_outliner = df['Survival Months'].mean()-66
right_outliner =df['Survival Months'].mean()+66


# In[56]:


print("left_outliner:",left_outliner,"|","right_outliner:",right_outliner)


# In[57]:


z_scoreS = (dm['Survival Months']-dm['Survival Months'].mean())/dm['Survival Months'].std()
threshold=2


# In[58]:


z_scoreS


# In[59]:


dmS=dm.drop(dm[abs(z_scoreS)>threshold].index)


# In[60]:


dmS


# In[61]:


plt.figure(figsize=(10,5))
plt.plot(df['Survival Months'],label='Survival Months')
plt.plot(dmS['Survival Months'],label='concentration Survival Months')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Survival Months Noise')
plt.legend()
plt.grid(True)
plt.show()


# In[62]:


#this survival month is droppinng early death


# In[63]:


dg=dm[['Age','Tumor Size','Survival Months']]


# In[64]:


dg


# In[67]:


dn=pd.concat([dmS['Survival Months'],dmT['Tumor Size'],dmA['Age']],axis=1)


# In[68]:


dn


# In[69]:


dn.describe()


# In[70]:


#we done with analyzing and removing noise from single numerical data.
#let analyze categorical data single column


# In[71]:


df['Race']


# In[72]:


df['Race'].unique()


# In[73]:


dfR=dm['Race']


# In[75]:


dfRR=dfR.value_counts().reset_index()


# In[76]:


dfRR.columns =['Race','count']


# In[77]:


dfRR


# In[84]:


percent=dm['Race'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(percent.values,labels=percent.index,autopct='%.1f%%',startangle=90,explode=[1,0.1,0],colors=['#ff9999','blue','yellow'])
plt.legend()
plt.title('RACE',fontsize=14)
plt.show()


# In[85]:


dm['Marital Status']


# In[87]:


dm['Marital Status'].unique()


# In[88]:


dfM=dm['Marital Status']


# In[89]:


dfMM=dfM.value_counts().reset_index()


# In[90]:


dfMM.columns =['Marital Status','count']


# In[91]:


dfMM


# In[96]:


plt.figure(figsize=(10,5))
plt.title('Marital Status', fontsize=20)
dm['Marital Status'].value_counts()[:10].plot(kind='barh', color='blue');


# In[99]:


dm['differentiate']


# In[100]:


dm['differentiate'].unique()


# In[101]:


dmd=dm['differentiate']


# In[102]:


dmdd=dmd.value_counts().reset_index()


# In[103]:


dmdd.columns=['differentiate','count']


# In[104]:


dmdd


# In[106]:


plt.figure(figsize=(10,5))
plt.title('differentiate', fontsize=20)
dm['differentiate'].value_counts()[:10].plot(kind='barh', color='pink');


# In[107]:


dm['Status']


# In[108]:


dm['Status'].unique()


# In[109]:


dms=dm['Status']


# In[110]:


dmss=dms.value_counts().reset_index()


# In[112]:


dmss.columns=['Status','counts']


# In[113]:


dmss


# In[121]:


percent=dm['Status'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(percent.values,labels=percent.index,autopct='%.1f%%',startangle=90,explode=[0.1,0],colors=['skyblue','gray'])
plt.legend()
plt.title('Alive Status',fontsize=14)
plt.show()


# In[122]:


#completed column analysis let consider multiple column analysis and visualization


# In[126]:


du=dm.copy()


# In[136]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in du.select_dtypes(include=[object]).columns:
    du[col+'_encodeed']=le.fit_transform(du[col])
du.drop(columns=du.select_dtypes(include=['object']).columns,inplace=True)
cm=du.corr()
cm     


# In[137]:


cm.columns.values


# In[146]:


cmm=cm[['Age', 'Tumor Size','Survival Months', 'Race_encodeed','Marital Status_encodeed','Status_encodeed','differentiate_encodeed']]


# In[147]:


cmm


# In[148]:


#from correlation of the table u can analyze the folowing
#age has no correlation
#tumor size has good correlation with tumor stage and  positive correlation with n stage
#no correlation above to observe


# In[151]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in du.select_dtypes(include=[object]).columns:
    du[col+'_encodeed']=le.fit_transform(du[col])
du.drop(columns=du.select_dtypes(include=['object']).columns,inplace=True)
cv=du.cov()
cv 


# In[165]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in du.select_dtypes(include=[object]).columns:
    du[col+'_encodeed']=le.fit_transform(du[col])
du.drop(columns=du.select_dtypes(include=['object']).columns,inplace=True)
du


# In[285]:





# In[169]:


du['T Stage _encodeed'].unique()


# In[179]:


df['Age'].unique()


# In[167]:


plt.figure(figsize=(8,6))
plt.scatter(df['Tumor Size'],du['T Stage _encodeed'],color=['blue'],label='tumor')
model = LinearRegression()
model.fit(df[['Tumor Size']],du['T Stage _encodeed'])
trend_line = model.predict(df[['Tumor Size']])
plt.plot(df['Tumor Size'],trend_line,color='red',label='Trend Line')
plt.xlabel('Tumor size')
plt.ylabel('t stage')
plt.legend()
plt.title('Tumor',fontsize=14)
plt.grid(True)
plt.show()


# In[214]:


plt.figure(figsize=(18,10))
counts = df.groupby(['Age','Status']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Age','Status'])
sns.countplot(x='Age',hue='Status',data=dmA,order=sorted_counts['Age'],dodge=True)


# In[212]:


sorted_counts


# In[ ]:





# In[ ]:





# In[223]:


count_Dead=sorted_counts[sorted_counts['Status']=='Dead']
count_Alive=sorted_counts[sorted_counts['Status']=='Alive']


# In[225]:


count_Dead


# In[226]:


count_Alive


# In[229]:


plt.figure(figsize=(16,10))
plt.plot(count_Alive['Age'],count_Alive['count'],lw=4,c='red',label='Alive')
plt.plot(count_Dead['Age'],count_Dead['count'],lw=4,c='black',label='Dead')
plt.legend()
plt.grid()
plt.title('Age statistic dead or alive', fontsize=18)
plt.show()


# In[231]:


plt.figure(figsize=(18,10))
counts = df.groupby(['Age','Survival Months']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Age','Survival Months'])
sorted_counts


# In[ ]:





# In[232]:


counts = df.groupby(['Race','Marital Status']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Race','Marital Status'])
sorted_counts


# In[236]:


plt.figure(figsize=(18,7))
counts = df.groupby(['Race','Marital Status']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Race','Marital Status'])
sns.countplot(x='Race',hue='Marital Status',data=df,order=sorted_counts['Race'],dodge=True)


# In[247]:


plt.figure(figsize=(16,10))
plt.scatter(df['Survival Months'],du['Status_encodeed'])
model = LinearRegression()
model.fit(df[['Survival Months']],du['Status_encodeed'])
trend_line = model.predict(df[['Survival Months']])
plt.plot(df['Survival Months'],trend_line,color='red',label='Trend Line')
plt.xlabel('survival month')
plt.ylabel('Status_encodeed')
plt.legend()
plt.title('survive',fontsize=14)
plt.grid(True)
plt.show()


# In[246]:


du['Status_encodeed'].unique()


# In[252]:


counts = df.groupby(['Tumor Size','T Stage ']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Tumor Size','T Stage '])
sorted_counts


# In[251]:





# In[254]:


plt.figure(figsize=(20,5))
counts = df.groupby(['Tumor Size','T Stage ']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['Tumor Size','T Stage '])
sns.countplot(x='Tumor Size',hue='T Stage ',data=df,order=sorted_counts['Tumor Size'],dodge=True)


# In[256]:


counts = df.groupby(['differentiate','Race']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['differentiate','Race'])
sorted_counts


# In[259]:


plt.figure(figsize=(22,5))
counts = df.groupby(['differentiate','Race']).size().reset_index(name='count')
sorted_counts=counts.sort_values(by=['differentiate','Race'])
sns.countplot(x='differentiate',hue='Race',data=df,order=sorted_counts['differentiate'],dodge=True)


# In[283]:


plt.figure(figsize=(15, 5), dpi=80) # Maximize the plot

plt.hist(df['Progesterone Status'], bins=35, histtype='stepfilled', align='mid', orientation='vertical', color='green')

plt.legend (['Progesterone Status'], loc = 'best')
plt.title ("Breast Cancer progesterone", fontsize=20)

plt.xlabel('Progesterone Status', fontsize=15)

plt.grid()
plt.show()


# In[275]:


plt.figure(figsize=(20,7))
plt.figure(figsize=(10, 6), dpi=80) # Maximize the plot

plt.hist(df['Regional Node Examined'], bins=35, histtype='stepfilled', align='mid', orientation='vertical', color='skyblue')

plt.legend (['Regional Node Examined'], loc = 'best')
plt.title ("Breast Cancer", fontsize=20)

plt.xlabel('Regional Node Examined', fontsize=15)


plt.grid()
plt.show()


# In[281]:


plt.figure(figsize=(18,7))
plt.figure(figsize=(10, 6), dpi=80) # Maximize the plot

plt.hist(df['Reginol Node Positive'], bins=35, histtype='stepfilled', align='mid', orientation='vertical', color='gray')

plt.legend (['Reginol Node Positive'], loc = 'best')
plt.title ("Breast Cancer regional positive node", fontsize=20)

plt.xlabel('Reginol Node Positive', fontsize=15)

plt.show()


# In[280]:


plt.figure(figsize=(18,7))
plt.hist(df['Survival Months'], bins=35, histtype='stepfilled', align='mid', orientation='vertical', color='gold')

plt.legend (['Survival Months'], loc = 'best')
plt.title ("Breast Cancer survival month", fontsize=20)

plt.xlabel('Survival Months', fontsize=15)

plt.grid()
plt.show()


# In[279]:


plt.figure(figsize=(20, 6), dpi=80) # Maximize the plot

plt.hist(df['Age'], 
         bins=35, 
         #range=None, 
         #density=False, 
         #weights=None, 
         #cumulative=False, 
         #bottom=None, 
         histtype='stepfilled', 
         align='mid', 
         orientation='vertical', 
         #rwidth=None, 
         #log=False, 
         color='purple', 
         #label=None, 
         #stacked=False, 
         #data=None
        )

plt.legend (['Age'], loc = 'best')
plt.title ("Age with Breast cancer", fontsize=20)

plt.xlabel('Age', fontsize=15)
#plt.xticks(rotation=90) # Rotating the x labels

#plt.savefig ('D:/IMT/3- Data Science/3- Data pre-processing (2)/7-Age_hist.png')

#plt.grid()
plt.show()


# In[286]:


du


# In[299]:


du.columns.values


# In[287]:


random_row=du.sample(n=1000)


# In[309]:


random_row.columns.values


# In[351]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X=random_row[['Age', 'Tumor Size', 'Regional Node Examined',
       'Reginol Node Positive', 'Survival Months', 'Race_encodeed',
       'Marital Status_encodeed', 'T Stage _encodeed', 'N Stage_encodeed',
       '6th Stage_encodeed', 'differentiate_encodeed', 'Grade_encodeed',
       'A Stage_encodeed', 'Estrogen Status_encodeed',
       'Progesterone Status_encodeed']]
y=random_row[ 'Status_encodeed']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# Create and train a Decision Tree classifier
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = tree_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')


# In[355]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X=random_row[['Age', 'Tumor Size', 'Regional Node Examined',
       'Reginol Node Positive', 'Survival Months', 'Race_encodeed',
       'Marital Status_encodeed', 'T Stage _encodeed', 'N Stage_encodeed',
       '6th Stage_encodeed', 'differentiate_encodeed', 'Grade_encodeed',
       'A Stage_encodeed', 'Estrogen Status_encodeed',
       'Progesterone Status_encodeed']]
y=random_row[ 'Status_encodeed']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')


# In[353]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X=random_row[['Age', 'Tumor Size', 'Regional Node Examined',
       'Reginol Node Positive', 'Survival Months', 'Race_encodeed',
       'Marital Status_encodeed', 'T Stage _encodeed', 'N Stage_encodeed',
       '6th Stage_encodeed', 'differentiate_encodeed', 'Grade_encodeed',
       'A Stage_encodeed', 'Estrogen Status_encodeed',
       'Progesterone Status_encodeed']]
y=random_row[ 'Status_encodeed']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# Define neural network architecture for logistic regression
model = Sequential([
    Dense(1, input_shape=(X_train.shape[1],), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# In[361]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X=random_row[['Age', 'Tumor Size', 'Regional Node Examined',
       'Reginol Node Positive', 'Survival Months', 'Race_encodeed',
       'Marital Status_encodeed', 'T Stage _encodeed', 'N Stage_encodeed',
       '6th Stage_encodeed', 'differentiate_encodeed', 'Grade_encodeed',
       'A Stage_encodeed', 'Estrogen Status_encodeed',
       'Progesterone Status_encodeed']]
y=random_row[ 'Status_encodeed']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

# Create and train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', random_state=100)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')


# In[362]:


win=du.drop([ 'Status_encodeed'],axis=1)


# In[363]:


win


# In[364]:


y_pred=svm_classifier.predict(win)


# In[365]:


y_pred


# In[366]:


win['prediction']=y_pred


# In[367]:


win


# In[368]:


merge=pd.concat([du[ 'Status_encodeed'],win['prediction']],axis=1)


# In[369]:


merge


# In[374]:


csv_file_path='merge.csv'
merge.to_csv(csv_file_path,index=False)
print("prediction saved to:",csv_file_path)


# In[ ]:





# In[ ]:




