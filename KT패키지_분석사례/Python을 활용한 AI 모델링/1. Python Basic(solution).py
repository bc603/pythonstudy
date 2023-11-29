#!/usr/bin/env python
# coding: utf-8

# # __1. 인덱싱(Indexing)__
# 
#   - 사전적으로는 무엇인가 "가르킨다"를 의미합니다. 
#   - 즉, 인덱싱(Indexing)은 특정한 값을 뽑아내는 역할을 합니다.
#   - 예시) 
#    x = Rome is not built in a day!
#    

# ![rome.png](attachment:c79c4dae-6c1a-418f-84a9-4ae0c5d38b81.png)

# In[1]:


x = "Rome is not built in a day!"


# In[2]:


x[0]


# In[3]:


x[-1]


# In[4]:


x[-2]


# In[5]:


x[-6]


# # __2. 슬라이싱(Slicing)__
# 
#   - 사전적으로는 무엇인가 "잘라낸다"를 의미합니다. 
#   - 아래 예시에서 원하는 단어를 뽑아내고자 할때 슬라이싱을 사용할 수 있습니다. 
#   - 예시) x = Rome is not built in a day!
#   - Sliicing의 형태 : x[시작번호 : 끝번호]
# 

# ![rome.png](attachment:c79c4dae-6c1a-418f-84a9-4ae0c5d38b81.png)

# In[6]:


x[0:4]


# In[7]:


# 끝 번호를 삭제 한다면 ..  
x[12:]


# In[8]:


# 시작번호를 삭제 한다면 ..  
x[:7]


# In[9]:


# 시작번호, 끝번호를 삭제 한다면 ..  
x[:]


# In[10]:


# 슬라이싱도 인덱싱 처럼 (-)기호를 사용할 수 있다.  
x[12:-7]


# # __3. Pandas 자료형__
# 
# - 자료형 : 프로그래밍을 할 때 쓰이는 숫자, 문자열 등 자료 형태로 사용하는 모든 것
# - 자료형은 데이터를 이해하는데 가장 기본이자 중요한 요소입니다. 
# 
# ### <b>3-1. 자료형의 종류</b>
#  - string (문자열) : 문자, 단어 등으로 구성된 문자들의 집합
#     ex)  "Hello World" , "123" , "a"
#  - integer : 숫자형 데이터 중 정수형
#     ex)  123, -234 
#  - float : 소수점이 포함된 숫자
#     ex)  1.2, -2.34

# ### <b>3-2. 리스트 자료형</b>
#   - 순서대로 정리 된 항목들을 담는 구조(특정 데이터를 반복적으로 처리하는데 특화 되어 있음
#   - 리스트는 [ ]로 감싸고 각 요소는 ,로 구분한다.

# #### <b>1) 리스트의 생성</b>
#  - 리스트명 = [데이터, 데이터, 데이터]
#  - 리스트명 = list(range(시작 값, 종료 값, 증가 or 감소)

# In[11]:


a=[1,2,3]
a


# In[12]:


b=list(range(1,10,2))
b


# #### <b>2) 리스트 관련 함수</b>
# - 리스트.append(요소) : 리스트의 끝에 요소 추가
# - 리스트.insert(인덱스, 요소) : 리스트의 특정 인덱스에 요소 추가
# - 리스트1.extend(리스트2) : 리스트1에 리스트2를 연결해 확장
# - 리스트1 + 리스트2 : 리스트1과 리스트2를 서로 병합
# - 리스트.remove(요소) : 리스트의 특정 값의 요소를 삭제
# - 리스트.count('요소') : 리스트 중 특정 값을 가진 요소의 개수를 카운트
# - 리스트.sort() : 리스트 내 요소를 오름차순으로 정렬
# - 리스트.pop(인덱스) : 리스트 중 특정 인덱스의 요소를 삭제

# In[13]:


a = ['red', 'green', 'blue']
a.append('yellow')
a


# In[14]:


a.insert(1, 'black')
a


# In[15]:


b = ['purple', 'white']
a.extend(b)
a


# In[16]:


c = a+b
print(c)


# In[17]:


d = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d.remove(90)
d


# In[1]:


list1 = ['a', 'bb', 'c', 'd','aaa', 'c', 'ddd', 'aaa',  'b', 'cc', 'd', 'aaa', ]
list1.count('aaa')


# In[2]:


list2 = [1, -7, 5, 8, 3, 9, 11, 13]
list2.sort()
list2


# In[3]:


list2.sort(reverse=True)
list2


# In[5]:


d = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d.pop(0) #remove는 value, pop은 인덱스
d

#특정 인덱스에 있는 값을 삭제


# ### <b>3-3. 튜플(Tuple)</b>
# - 리스트와 거의 유사하고 사용법도 동일함
# - 라스트는 [ ] 으로 둘러싸지만 튜플은 ( )으로 둘러싼다.
# - 튜플에서는 리스트와 달리 요소들의 수정과 추가 불가

# #### <b>1) 튜플 생성</b>
# - 튜플명 = ( 데이터, 데이터, 데이터, ... )
# - 튜플명 = 데이터, 데이터, 데이터, ...
# 
# 괄호가 있는 형태도 있고, 괄호가 없는 형태도 있다

# In[7]:


t1 = ('banana', ' apple', 'kiwi')
t1


# In[8]:


t2 = 'banana', 'apple', 'kiwi'
t2


# In[9]:


# 튜플은 변환이 불가능 하기 때문에 에러 발생
t2[0] = 'watermelon'


# #### <b>2) 튜플의 장점</b>
# - 튜플은 반복 처리 시 리스트에 비해 근소하게나마 빠르다. (연산량이 많은 경우 약간의 성능 향상)
# - 튜플은 요소를 변경할 수 없는 특성으로 인해 딕셔너리의 키로 사용할 수 있다. (리스트는 사용 불가)
# - 튜플은 요소를 변경할 수 없기 때문에 보안을 요하는 데이터를 보호하는 장점을 가지게 된다.

# ### <b>3-4. 딕셔너리(Dictionary)</b>
# - 사전이 단어와 뜻으로 구성되듯, 키(key)와 값(Value)의 쌍으로 갖는 자료형

# #### <b>1) 딕셔너리 생성</b>
# - 딕셔너리명 = {key1:value1, key2:value2, key3:value3, ...}
# - 딕셔너리 key 값으로 튜플은 사용 가능 하지만 리스트는 사용불가 

# In[10]:


members = {'name':'안지영', 'age':30, 'email':'jiyoung@korea.com'}
members


# #### <b>2) 딕셔너리 관련 함수</b>
#   - keys : 딕셔너리에 key만 모아서 반환
#   - vlaues : 딕셔너리에 value만 모아서 반환
#   - items : key, value쌍을 튜플로 묶은 값을 반환
#   - clear : key,vallue 모두 지우기
#   - get : key로 value 얻기
#   - in : 해당 key가 딕셔너리 안에 있는지 조사하고자 하는 경우 사용

# In[13]:


members.keys()
type(members.keys())


# In[14]:


members.values()


# In[15]:


members.items()


# In[16]:


members.get('name')


# In[18]:


'name' in members
#name이라는 key가 members안에 있는지 확인


# In[19]:


'birth' in members


# In[20]:


members.clear()
members


# ### <b>3-5. List, Tuple, Dictionary차이</b>
# 
# | 구분 | 설명 |
# |------|---|
# |리스트(List) |순서대로 정리된 항목들을 담는 자료구조|
# |튜플(Tuple)|리스트와 비슷하지만 수정이 불가능하여 비정적인 객체를 담는데 사용|
# |딕셔너리(Dictionary)|key와 value로 구성되는 자료구조|
