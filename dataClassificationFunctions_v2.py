import json
import itertools
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from functools import reduce

def dataPercentMatch(df):
    d = []
    pattern_desc = {'NN':'this is string','CD':'this is numeric','JJ':'this is dates'}
    for i in df.columns:
        #s = df[i].tolist()
        l = [nltk.pos_tag(nltk.word_tokenize(str(i))) for i in df[i].tolist()]

        temp_df = pd.DataFrame(l)
        temp_df['names'] = [x for (x,y) in temp_df.iloc[:,0]]
        temp_df['tag'] = [y for (x, y) in temp_df.iloc[:,0]]
        #print(temp_df)

        count_pc = temp_df['tag'].value_counts(normalize=True) * 100
        values = temp_df['tag'].value_counts().index.tolist()
        d.append((i,count_pc.max(),values[0]))

    f_df = pd.DataFrame(d,columns=['column_names','percent_match','pattern'])
    #f_df.loc[:,'pattern_desc'] = f_df['pattern'].map(pattern_desc)
    f_df['pattern_desc'] = f_df['pattern'].map(pattern_desc)
    f_df = f_df[['column_names','percent_match','pattern_desc']] #comment this out for debugging patterns

    return (f_df)


# Function to get the highest rank
def performRanking(dataframe,type):
    dataframe['rank'] = dataframe.groupby('column_names')['rating'].rank(method='first')
    filter_highest_rank = dataframe['rank'] == 1.0

    df_highest_rank = dataframe[filter_highest_rank][['column_names', type,'pattern','rating']]
    df_highest_rank = df_highest_rank .drop(['rating'], axis=1)

    return df_highest_rank

# Function for PII classification
def getPIIClassification(type,json_object, column_list):
    c1_df = []
    c_personal = json_object['Personal Identification'][0]
    p1_list = [c_personal[value] for key, value in enumerate(c_personal)]
    personal = list(itertools.chain.from_iterable(p1_list))
    for value in column_list:
        c_p = [(value, item, "Yes", 1) for item in personal if (re.search(item, str(value), re.IGNORECASE))]

        if not bool(c_p):
            c1_df.append((value, "no pattern", "No", 2))

        c1_df = c_p + c1_df


    c2_df = pd.DataFrame(c1_df, columns=['column_names', 'pattern', type, "rating"])
    c2_df['rank'] = c2_df.groupby('column_names')['rating'].rank(method='first')

    filter_highest_rank = c2_df['rank'] == 1.0
    o_df = c2_df[filter_highest_rank][['column_names', type,'pattern','rating']]

    return o_df

# Function for Data labelling classification
def getLabellingClassification(type,rating,json_object, column_list):
    label1 = []
    c1_df = []
    for key in json_object:
        c_label = [json_object[key][v] for k, v in enumerate(json_object[key])]
        label = list(itertools.chain.from_iterable(c_label))
        label1.append(label)

    label2 = list(itertools.chain.from_iterable(label1))
    for value in column_list:
        c_label = [(value, item, type, rating) for item in label2 if (re.search(item, str(value), re.IGNORECASE))]

        if not bool(c_label):
            c1_df.append((value,"zno pattern","Unrestricted",4))

        c1_df = c_label + c1_df


    c2_df = pd.DataFrame(c1_df, columns=['column_names', 'pattern', 'Label',"rating"])

    return c2_df

# Function for CIA classification
def getCIAClassification(type,json_object, column_list):
    c1_df = []
    c_high = json_object[type][0]['High']
    c_medium = json_object[type][0]['Medium']

    c1_high = [c_high[value] for key, value in enumerate(c_high)]
    c1_medium = [c_medium[value] for key, value in enumerate(c_medium)]

    high = list(itertools.chain.from_iterable(c1_high))
    medium = list(itertools.chain.from_iterable(c1_medium))

    for value in column_list:
        c_h = [(value,item,"High",1) for item in high if (re.search(item, str(value), re.IGNORECASE))]
        c_m = [(value,item,"Medium",2) for item in medium if (re.search(item, str(value), re.IGNORECASE))]

        if not bool(c_h + c_m):
            c1_df.append((value,"no pattern","Low",3))

        c1_df = c_h + c_m + c1_df


    c2_df = pd.DataFrame(c1_df, columns=['column_names', 'pattern', type,'rating'])

    o_df = performRanking(c2_df,type)

    return o_df

# Parent data classification function
def getDataClassification(column_list):
    with open('confidential_patterns.json', 'r',encoding='utf-8') as json_file:
        confidential_d = json.load(json_file)
    with open('integrity2_patterns.json', 'r',encoding='utf-8') as json_file:
        integrity_d = json.load(json_file)
    with open('availability2_patterns.json', 'r',encoding='utf-8') as json_file:
        availability_d = json.load(json_file)

    with open('personal_identification.json', 'r',encoding='utf-8') as json_file:
        personal_d = json.load(json_file)

    with open('secret_label_MASTER_v1.json', 'r',encoding='utf-8') as json_file:
        secret_d = json.load(json_file)
    with open('res_internal_label_MASTER_v1.json', 'r',encoding='utf-8') as json_file:
        res_internal_d = json.load(json_file)
    with open('res_external_label_MASTER_v1.json', 'r',encoding='utf-8') as json_file:
        res_external_d = json.load(json_file)

    # CIA Classification
    c_df = getCIAClassification('Confidential', confidential_d, column_list)
    i_df = getCIAClassification('Integrity', integrity_d, column_list)
    a_df = getCIAClassification('Availability', availability_d, column_list)

    #PII Classification
    p_df = getPIIClassification('Personal Identification', personal_d, column_list)

    #Labelling Classification
    secret_df = getLabellingClassification('Secret', 1,secret_d, column_list)
    res_int_df = getLabellingClassification('Restricted-Internal', 2, res_internal_d, column_list)
    res_ext_df = getLabellingClassification('Restricted-External', 3, res_external_d, column_list)
    label_df = pd.concat([secret_df,res_int_df, res_ext_df],axis=0).reset_index(drop=True)
    label_highest_rank = performRanking(label_df,'Label')

    # Merge all the data frames
    data_frames = [c_df,i_df,a_df,p_df,label_highest_rank]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['column_names'],how='left'),data_frames)

    df_merged.columns = ['column_names', 'Confidential', 'Confidential_pattern', 'Integrity', 'Integrity_pattern',
       'Availability', 'Availability_pattern', 'Personal Identification', 'PII_pattern',
       'rating', 'Label', 'Label_pattern']

    df_merged = df_merged[['column_names','Label','Confidential','Availability','Personal Identification']] #comment this out if you do not want the pattern to be displayed

    return df_merged

def getTokensForDataClassification(r):
    word_tokens = word_tokenize(r)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.pos_tag(filtered_sentence)
    nouns = [word for word, pos in tagged if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    m = getDataClassification(nouns)
    #print(m) # comment this out for debugging

    #n = m.drop(['column_names'],axis=1).replace(['High', 'Medium', 'Low','Secret','Restricted-Internal','Restricted-External','Unrestricted','Yes','No'], [1, 2, 3, 4,5,6,7,8,9])
    n = m.drop(['column_names'], axis=1).replace(['High', 'Medium', 'Low', 'Secret', 'Restricted-Internal', 'Restricted-External', 'Unrestricted', 'Yes', 'No'], [1, 2, 3, 4, 5, 6, 7, 8, 9]).min()
    j = n.replace([1, 2, 3, 4,5,6,7,8,9],['High', 'Medium', 'Low','Secret','Restricted-Internal','Restricted-External','Unrestricted','Yes','No'])

    return(j)

def classifyDataNLP(df):
    class_byDesc = df['desc'].apply(getTokensForDataClassification)
    class_byDesc['column_names'] = df['column_name']

    column_list = df['column_name'].tolist()
    class_byColumns = getDataClassification(column_list)

    label_df = pd.concat([class_byDesc, class_byColumns], axis=0).reset_index(drop=True)

    label_df = label_df.replace(
        ['High', 'Medium', 'Low', 'Secret', 'Restricted-Internal', 'Restricted-External', 'Unrestricted', 'Yes', 'No'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9])
    label_df = label_df.groupby(['column_names']).min()
    label_df = label_df.replace([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                ['High', 'Medium', 'Low', 'Secret', 'Restricted-Internal', 'Restricted-External',
                                 'Unrestricted', 'Yes', 'No']).reset_index()

    return(label_df)
