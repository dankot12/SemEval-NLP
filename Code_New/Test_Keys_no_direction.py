

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


# removing the tags from the sentences

data = []
headers = ["Category"]
count = 0
row = []
for i in lines:

    if count == 1:
        if i != 'Other':
            row += [i[:-7]]
        else:
            row += [i]

    count += 1
    if count == 4:
        count = 0
        data.append(row)
        row = []




# create a data frame from the modified sentences

datadf = pd.DataFrame(data, columns=headers)


# write the dataframe into a csv file

datadf.to_csv('CSV Files/Test_Keys_no_direction.csv', sep=',', encoding='utf-8', index = False)
