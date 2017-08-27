
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup

# opening the raw train file provided
lines = []
i = 1
for line in open('Text Files/TRAIN_FILE.TXT'):
    line = line.strip(' \t\n\r')
    lines.append(line)

# removing the tags from the sentences
data = []
headers = ["Sentence","Category"]
count = 0
j = 1
for i in lines:
    if count == 0:
        row = []
        soup = BeautifulSoup(i,'html.parser')
        link1 = soup.e1
        link2 = soup.e2
        e1 = link1.get_text()
        e2 = link2.get_text()
        i = i.replace(str(j)+'\t','')
        i = i.replace('<e1>','')
        i = i.replace('</e1>','')
        i = i.replace('<e2>','')
        i = i.replace('</e2>','')
        row += [i]
        j += 1
    if count == 1:
        row += [i]
    count += 1
    
    if count == 4:
        count = 0
        data.append(row)
    
# create a data frame from the modified sentences

datadf = pd.DataFrame(data, columns=headers)
# write the dataframe into a csv file
datadf.to_csv('CSV Files/Train_data_direction.csv', sep=',', encoding='utf-8', index = False)

