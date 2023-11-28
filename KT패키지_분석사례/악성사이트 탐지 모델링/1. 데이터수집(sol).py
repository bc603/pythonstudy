#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install bs4


# In[2]:


pip install openpyxl


# In[4]:


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

# In[8]:


filename ='Feature Website.xlsx'
#xlrd 지원불가로 openpyxl 사용할 수 있도록 변경
df = pd.read_excel(filename, engine='openpyxl')


# In[9]:


df.info()


# In[12]:


df


# In[ ]:





# ### <b>Q2. html 에서 \<script>...\</script> 길이 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[22]:


def html_script_characters(soup):
    html_len = str(soup.script)
    return float(len(html_len.replace(' ', '')))


# In[20]:


script_len = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    script_len.append(html_script_characters(soup))

df['html_script_characters'] = script_len


# In[21]:


df


# In[23]:


df.describe()


# ### <b>Q3. html에서 공백 수 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[24]:


def html_num_whitespace(soup):
    try:
        # soup > body > text > count
        NullCount = soup.body.text.count(' ')
        return float(NullCount)
    except:
        return 0.0


# In[25]:


num_whitespace = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    num_whitespace.append(html_num_whitespace(soup))

df['html_num_whitespace'] = num_whitespace


# In[26]:


df


# In[27]:


df.describe()


# In[ ]:





# ### <b>Q4. html 에서 body 길이 계산</b>
# ---
# - BeautifulSoup으로 html소스를 python 객체로 변환
# - 함수로 구현하기
# - float으로 return 받기

# In[38]:


def html_num_characters(soup):
    try:
        #soup > body > text
        bodyLen = len(soup.body.text)
        return float(bodyLen)
    except:
        return 0.0


# In[39]:


html_body = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    html_body.append(html_num_characters(soup))

df['html_body_len'] = html_body


# In[40]:


df


# In[41]:


df.describe()


# ### <b> Q5. script 에서 src, href 속성을 가진 태그수</b>
# 1. BeautifulSoup으로 html소스를 python 객체로 변환
# 2. 함수로 구현하기
# 3. float으로 return 받기

# In[42]:


def html_link_in_script(soup):
    numOfLinks = len(soup.findAll('script', {"src": True}))
    numOfLinks += len(soup.findAll('script', {"href": True}))
    return float(numOfLinks)


# In[43]:


html_script_link_num = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    html_script_link_num.append(html_link_in_script(soup))

df['html_script_link_num'] = html_script_link_num


# In[44]:


df


# In[45]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




