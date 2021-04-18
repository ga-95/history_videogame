
import glob
import pandas as pd
import sys
import re

try:
    path=sys.argv[1]
    if path[-1]!="\\" and path[-1]!="/":
        if "/" in path:
            path=path+"/"
        if "\\" in path:
            path=path+"\\"

except:
    path=''

def clean(series):
    series= series.str.lower()
    series=series.dropna()
    for i in "“#‼$%&!'-★”’()*+,-./:;<=>?@[\\]🤣😂^'🤦🔴⚠_`{|}\n\t\"":
        series = series.str.replace(i, " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    series = series.str.replace("  ", "#")
    series = series.str.replace("#", " ")
    print(len(series.str.contains("  ")))
    return series

def remove_space(string):
    string = string.strip()
    string= string.lstrip()
    string = string.rstrip()
    if '  ' in string:
        while '  ' in string:
            string = string.replace('  ', ' ')
    return string

print (path)
files=glob.glob(path+"*_keywords.xlsx")
files=glob.glob(path+"*_keywords.csv")

print (files)
dataframe_keywords=pd.read_excel("file_keyword.xlsx")
print (dataframe_keywords.columns)
print(dataframe_keywords)
for file in files:
    print (file)
    nomefile=file.split(".")[0]
    luogo=nomefile.split("_")[0].lower()
    print(luogo)
    nomefile_output = luogo + "_keywords_processate.csv"
    df=pd.read_csv(file, sep=",")
    print(df.columns)
    #colonna_keywords = df["keywords"]
    #keywords = dataframe_keywords.iloc[:, lambda df: dataframe_keywords.columns.str.contains(luogo, case=False)].head()
    keywords = dataframe_keywords[luogo].dropna()
    #print (keywords)
    totale=0
    buffer=pd.DataFrame()
    df["Message"] = df["Message"].astype(str, int)
    df["Message"]= clean(df["Message"])
    message= clean(df["Message"])
    for msg in df["Message"]:
        msg= remove_space(msg)
        #print(msg)

    for index, row in df.iterrows():
        counter=0
        if index%10000==0:
            print (luogo,index,len(df),totale)
        #rec=row["Message"]
        rec = row["Message"]#.strip().replace("\r\n"," ").replace("\t"," ")
        #print(rec)
        for k in keywords:
            if " "+k+" " in rec:
                counter += 1
            else:
                pass
        if counter>0:
            print (rec)
            buffer=buffer.append(row)
            totale+=1
            print ("trovato n° ",totale)
        df.at[index, "keywords"] = counter
    buffer=buffer[['Page Name', 'User Name', 'Facebook Id', 'Likes at Posting',
       'Followers at Posting', 'Post Created', 'Post Created Date',
       'Post Created Time', 'Type', 'Total Interactions', 'Likes', 'Comments',
       'Shares', 'Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care',
       'Video Share Status', 'Is Video Owner?', 'Post Views', 'Total Views',
       'Total Views For All Crossposts', 'Video Length', 'URL', 'Message',
       'Link', 'Final Link', 'Image Text', 'Link Text', 'Description',
       'Sponsor Id', 'Sponsor Name', 'Sponsor Category',
       'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )']]
    buffer.to_csv(luogo+"_positivi.csv",sep="\t")

    df = df[['Page Name', 'User Name', 'Facebook Id', 'Likes at Posting',
       'Followers at Posting', 'Post Created', 'Post Created Date',
       'Post Created Time', 'Type', 'Total Interactions', 'Likes', 'Comments',
       'Shares', 'Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care',
       'Video Share Status', 'Is Video Owner?', 'Post Views', 'Total Views',
       'Total Views For All Crossposts', 'Video Length', 'URL', 'Message',
       'Link', 'Final Link', 'Image Text', 'Link Text', 'Description',
       'Sponsor Id', 'Sponsor Name', 'Sponsor Category',
       'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )']]
    df.to_csv(nomefile_output.split(".")[0] + "_keywords_processate.csv", sep="\t", index=False)
