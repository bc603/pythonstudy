#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install bs4
#BeuatifulSoup 사용


# In[2]:


pip install openpyxl
#엑셀을 사용하기 위한 엑셀 라이브러리


# In[3]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from aicentro.session import Session
sacp_session = Session(verify=False)
from aicentro.framework.framework import BaseFramework as SacpFrm
sacp_framework = SacpFrm(session=sacp_session)


# ### <b>Q1. 데이터 불러오기</b>
# ---
# - Feature Website.xlsx 데이터 불러오기
# - 데이터 둘러보기

# In[4]:


filename ='Feature Website.xlsx'
#xlrd 지원불가로 openpyxl 사용할 수 있도록 변경
df = pd.read_excel(filename, engine='openpyxl')


# In[5]:


df.info()
#데이터 불러오기


# In[6]:


df
#데이터프레임 데이터 확인


# In[ ]:





# ### <b>Q2. html 에서 \<script>...\</script> 길이 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[7]:


def html_script_characters(soup):
    # soup > script
    html_len = str(soup.script)
    return float(len(html_len.replace(' ', '')))


# In[8]:


script_len = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    script_len.append(html_script_characters(soup))

df['html_script_characters'] = script_len


# In[9]:


df


# In[10]:


df.describe()


# ### <b>Q3. html에서 공백 수 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[18]:


def html_num_whitespace(soup):
    try:
        # soup > body > text > count
        NullCount = soup.body.text.count(' ') #작성하는 부분
        return float(NullCount)
    except:
        return 0.0


# In[19]:


num_whitespace = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser') #작성하는 부분
    num_whitespace.append(html_num_whitespace(soup))

df['html_num_whitespace'] = num_whitespace


# In[31]:


df # 동작한 결과를 확인


# In[32]:


df.describe() #데이터에 대해서 확인


# In[ ]:





# ### <b>Q4. html 에서 body 길이 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[21]:


def html_num_characters(soup):
    try:
        #soup > body > text
        bodyLen = len(soup.body.text) #구현하는 부분
        return float(bodyLen)
    except:
        return 0.0


# In[23]:


html_body = [] #리스트 선언

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser') #구현하는 부분
    html_body.append(html_num_characters(soup))

df['html_body_len'] = html_body


# In[24]:


df


# In[25]:


df.describe()


# ### <b> Q5. script 에서 src, href 속성을 가진 태그수</b>
# 1. BeautifulSoup으로 html소스를 python 객체로 변환
# 2. 함수로 구현하기
# 3. float으로 return 받기

# In[27]:


def html_link_in_script(soup):
    numOfLinks = len(soup.findAll('script', {"src": True}))
    numOfLinks += len(soup.findAll('script', {"href": True})) #작성하는 곳
    return float(numOfLinks)


# In[28]:


html_script_link_num = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    html_script_link_num.append(html_link_in_script(soup))

df['html_script_link_num'] = html_script_link_num


# In[29]:


df


# In[30]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




