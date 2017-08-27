
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup


# opening the raw test file provided
lines = []
i = 1
for line in open('Text Files/TEST_FILE_FULL.TXT'):
    line = line.strip(' \t\n\r')
    lines.append(line)


data = []
headers = ["Sentence"]
count = 0
j = 8001
for i in lines:
    if count == 0:
        row = []
        i = i.replace(str(j)+'\t','')
        i = i.replace('<e1>','')
        i = i.replace('</e1>','')
        i = i.replace('<e2>','')
        i = i.replace('</e2>','')
        row += [i]
        j += 1
    
    count += 1
    if count == 4:
        count = 0
        data.append(row)
    
    


# create a data frame from the modified sentences

datadf = pd.DataFrame(data, columns=headers)


# write the dataframe into a csv file

datadf.to_csv('CSV Files/Test_data.csv', sep=',', encoding='utf-8', index = False)

