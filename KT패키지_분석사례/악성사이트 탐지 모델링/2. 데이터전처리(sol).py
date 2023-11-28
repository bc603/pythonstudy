#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np

from aicentro.session import Session
sacp_session = Session(verify=False)
from aicentro.framework.framework import BaseFramework as SacpFrm
sacp_framework = SacpFrm(session=sacp_session)


# ## <b>학습데이터 불러오기</b>
# ---

# In[55]:


df = pd.read_csv('TrainData.csv',delimiter=',')


# In[56]:


df


# In[57]:


df.info()


# In[58]:


df.describe()


# ## <b>데이터 탐색/정제</b>
# 

# ### <b>Q1. 중복 데이터 제거</b>
# ---
#  - 중복된 행데이터 삭제

# In[59]:


df.info()


# In[60]:


# 중복 데이터 제거 : drop_duplicates()
df=df.drop_duplicates()


# In[61]:


df.info()


# ### <b>Q2. 텍스트와 범주형 특성 처리</b>
# ---
#  - replace() 함수를 이용한 텍스트와 범주형 특성 처리
#  - 참고사이트 : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
# 
# #### <b>replace() 함수 사용 예</b>
# ---

# In[62]:


import pandas as pd

df_ex = pd.DataFrame({'name': ['Alice','Bob','Charlie','Dave','Ellen','Frank'],
                   'age': [24,42,18,68,24,30],
                   'state': ['NY','CA','CA','TX','CA','NY'],
                   'point': [64,24,70,70,88,57]}
                  )

print(df_ex)


# In[63]:


df_ex['state'].replace({'CA':'California','NY':'NewYork'}, inplace=True)
print(df_ex)


# #### <b>텍스트와 범주형 특성 처리 실습</b>
# ---

# In[64]:


# unique() 유일한 값 확인
df['Result_v1'].unique()


# In[65]:


# replace() 함수 사용
# 'benign'=1,'malicious'=-1 처리
df['Result_v1'].replace({'benign':1,'malicious':-1}, inplace=True)


# In[66]:


df['Result_v1'].unique()


# In[ ]:





# ### <b>Q3. 결측치 제거</b>
# ---

# In[67]:


df.info()


# In[68]:


# 결측치 제거
# dropna()

df = df.dropna(axis=0)


# In[69]:


df.info()


# ### <b>Q4. 데이터 탐색을 통한 불필요한 칼럼 제거</b>
# ---
# - ex) 그래프, 데이터 상관관계 corr()등 활용
# - corr() 메서드 : 모든 특성 간의 표준 상관계수(피어슨의 R)
#   - 확률론과 통계학에서 두 변수간에 어떤 선형적 또는 비선형적 관계를 갖고 있는지를 분석하는 방법
#   - r 값은 X 와 Y 가 완전히 동일하면 +1, 전혀 다르면 0, 반대방향으로 완전히 동일 하면 –1 
# - scatter 그래프를 활용한 데이터 분석

# #### <b>corr() 함수 사용 예</b> 
# ---

# In[70]:


from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df_ex = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_ex['target'] = iris.target


# In[71]:


df_ex


# In[72]:


print(iris.DESCR)


# In[73]:


df_ex.corr()


# In[74]:


df_ex.corr()['target'].sort_values(ascending=False)


# #### <b>scatter 그래프를 활용한 데이터 분석 예</b>
# ---

# In[75]:


import matplotlib.pyplot as plt


# In[76]:


df_ex['color'] = df_ex['target'].map({0:"red", 1:"blue", 2:"green"})


# In[77]:


df_ex


# In[78]:


y_list = df_ex.columns

for i in range(0, len(y_list)):
    df_ex.plot(kind='scatter',x='target',y=y_list[i],s=30, c=df_ex['color'])
    plt.title('Scatter', fontsize=20)
    plt.xlabel('target')
    plt.ylabel(y_list[i])
    plt.show()


# In[ ]:





# In[ ]:





# #### <b>corr() 상관관계를 활용한 데이터 분석 실습</b>
# ---

# In[79]:


df.corr()


# In[80]:


df.corr()['Result_v1'].sort_values(ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# #### <b>scatter 그래프를 활용한 데이터 분석 실습</b>
# ---

# In[81]:


import matplotlib.pyplot as plt


# In[ ]:


# 1 = 'blue',-1 = 'red' 처리
df['color'] = df['Result_v1'].map({1:"blue", -1:"red"})


# df

# In[83]:


y_list = df.columns

for i in range(0, len(y_list)):
    df.plot(kind='scatter',x='Result_v1',y=y_list[i],s=30, c=df['color'])
    plt.title('Scatter Benign Malicious', fontsize=20)
    plt.xlabel('Result_v1')
    plt.ylabel(y_list[i])
    plt.show()


# In[ ]:





# #### <b>불필요한 컬럼 제거</b>
# ---

# In[84]:


df.drop(columns=["url_chinese_present","html_num_tags('applet')"],inplace=True)


# In[85]:


df.info()


# In[ ]:





# ## Q5. train_test_split을 이용하여, train_x, val_x, train_y, val_y로 데이터 분리
# ---
# 
# - test_size = 0.3
# - random_state = 2021

# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X = df.iloc[:,0:len(df.columns)-1].values
y = df.iloc[:,len(df.columns)-1].values


# In[52]:


# train_test_split 사용
train_x, val_x, train_y, val_y = train_test_split(X, y,test_size=0.3,random_state=2021) 


# In[53]:


train_x.shape, val_x.shape, train_y.shape, val_y.shape


# In[ ]:




