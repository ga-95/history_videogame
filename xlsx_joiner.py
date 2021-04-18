import pandas as pd
import glob

all_data = pd.DataFrame()
for f in glob.glob("*.xlsx"):
    print(f)
    df = pd.read_excel(f)
    print(df)
    all_data = all_data.append(df,ignore_index=True)

print(all_data)
all_data.to_csv("wf_aggregato.csv")