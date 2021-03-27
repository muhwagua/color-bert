import os 
import glob
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

#xml 덤프 파일 읽기 
os.chdir("color-bert/blogs")
xml_l = glob.glob('*.xml') 
error=0
text_dump=[]
for xml in xml_l:
    try:
        with open(f'{xml}', 'r') as f:
            bs4 = BeautifulSoup(f, features="html.parser")
            posts = bs4.find_all('post')
            for post in posts:
                text_dump += post.text.split('.')
    except:
        error+=1
print(f'{error} files skipped')

#간단히 전처리 하기 
print(len(text_dump))
text_corpus =[]
for n in range(len(text_dump)):
    sent = text_dump[n].strip()
    if not sent == '':
        text_corpus.append(sent)
print(len(text_corpus))

#색깔 단어 있는 문장 뽑기 1
color_list = ['red','orange','yellow','green','blue','purple','brown','white','black','pink','lime','gray','violet','cyan','magenta','khaki']
color_sent = []
for sentence in text_corpus:
    if color_list[0] in sentence:
        color_sent.append(sentence)
        
    elif color_list[1] in sentence:
        color_sent.append(sentence)
        
    elif color_list[2] in sentence:
        color_sent.append(sentence)
        
    elif color_list[3] in sentence:
        color_sent.append(sentence)
        
    elif color_list[4] in sentence:
        color_sent.append(sentence)
        
    elif color_list[5] in sentence:
        color_sent.append(sentence)
        
    elif color_list[6] in sentence:
        color_sent.append(sentence)
        
    elif color_list[7] in sentence:
        color_sent.append(sentence)
        
    elif color_list[8] in sentence:
        color_sent.append(sentence)
        
    elif color_list[9] in sentence:
        color_sent.append(sentence)
        
    elif color_list[10] in sentence:
        color_sent.append(sentence)
        
    elif color_list[11] in sentence:
        color_sent.append(sentence)
        
    elif color_list[12] in sentence:
        color_sent.append(sentence)
        
    elif color_list[13] in sentence:
        color_sent.append(sentence)        
        
    elif color_list[14] in sentence:
        color_sent.append(sentence)
        
    elif color_list[15] in sentence:
        color_sent.append(sentence)  

print(len(color_sent))

#색깔 단어 있는 문장 뽑기 2
color_text = []
chromatic_color = wn.synsets('red')[0].hypernyms()
n=0
error=0

for sent in color_sent:
    for word in sent.split(): 
        if len(wn.synsets(word)) != 0 and wn.synsets(word)[0].hypernyms() == chromatic_color:
                color_text.append(sent)
  
    n+=1
    if n % 10000 == 0:
        print(n)
        
color_text = list(set(color_text))

os.chdir("/content/color-bert")
with open('blog_dump.txt', 'w') as f:
    for item in c:
        f.write("%s\n" % item)
