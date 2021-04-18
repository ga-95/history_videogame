
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
    for i in "“#‼$%&!'♥-★”’()*+,-./:;<=>?@[\]🤣😂^'🤦🔴⚠_`{|}\n\t\"":
        series = series.str.replace(i, " ")
    while len(series.str.contains("  ")) > 0:
        print(len(series.str.contains("  ")))
        series = series.str.replace("  ", " ")
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
    df=pd.read_excel(file)
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
        #print(msg)

    for index, row in df.iterrows():
        group_name = df.at[index, 'Group Name']
        fb_id = df.at[index, 'Facebook Id']
        likes_at_post = df.at[index, 'Likes at Posting']
        followers = df.at[index, 'Followers at Posting']
        created = df.at[index, 'Created']
        type = df.at[index, 'Type']
        likes = df.at[index, 'Likes']
        comments = df.at[index, 'Comments']
        shares = df.at[index, 'Shares']
        love = df.at[index, 'Love']
        wow = df.at[index, 'Wow']
        laugh = df.at[index, 'Haha']
        sad = df.at[index, 'Sad']
        angry = df.at[index, 'Angry']
        care = df.at[index, 'Care']
        video_share = df.at[index, 'Video Share Status']
        post_view = df.at[index, 'Post Views']
        total_views = df.at[index, 'Total Views']
        tot_view_all = df.at[index, 'Total Views For All Crossposts']
        video_length = df.at[index, 'Video Length']
        url = df.at[index, 'URL']
        message = df.at[index, 'Message']
        link = df.at[index, 'Link']
        final_link = df.at[index, 'Final Link']
        image_text = df.at[index, 'Image Text']
        link_text = df.at[index, 'Link Text']
        description = df.at[index, 'Description']
        sponsor_id = df.at[index, 'Sponsor Id']
        sponsor_name = df.at[index, 'Sponsor Name']
        interaction = df.at[index, 'Total Interactions']
        score = df.at[
            index, 'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )']

        counter=0
        if index%10000==0:
            print (luogo,index,len(df),totale)
        #rec=row["Message"]
        rec = row["Message"]#.strip().replace("\r\n"," ").replace("\t"," ")
        #print(rec)
        for msg in df["Message"]:
            msg = remove_space(msg)
            for k in keywords:
                if " "+k+" " in msg:
                    counter += 1
                else:
                    pass
            if counter>0:
                print (msg)
                buffer=buffer.append()
                totale+=1
                print ("trovato n° ",totale)
        df.at[index, "keywords"] = counter
    buffer=buffer[['Page Name', 'User Name', 'Facebook Id', 'Likes at Posting',
       'Followers at Posting', 'Created', 'Type', 'Likes', 'Comments',
       'Shares', 'Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care',
       'Video Share Status', 'Post Views', 'Total Views',
       'Total Views For All Crossposts', 'Video Length', 'URL', 'Message',
       'Link', 'Final Link', 'Image Text', 'Link Text', 'Description',
       'Sponsor Id', 'Sponsor Name', 'Total Interactions',
       'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )']]
    buffer.to_csv(luogo+"_positivi.csv",sep="\t")

    df = df[['Page Name', 'User Name', 'Facebook Id', 'Likes at Posting',
       'Followers at Posting', 'Created', 'Type', 'Likes', 'Comments',
       'Shares', 'Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care',
       'Video Share Status', 'Post Views', 'Total Views',
       'Total Views For All Crossposts', 'Video Length', 'URL', 'Message',
       'Link', 'Final Link', 'Image Text', 'Link Text', 'Description',
       'Sponsor Id', 'Sponsor Name', 'Total Interactions',
       'Overperforming Score (weighted  —  Likes 1x Shares 1x Comments 1x Love 1x Wow 1x Haha 1x Sad 1x Angry 1x Care 1x )']]
    df.to_csv(nomefile_output.split(".")[0] + "_keywords_processate.csv", sep="\t", index=False)