import os
import pandas as pd

df = pd.read_csv(r"E:\glptc\toy_test\process_list_autogen.csv") # 这个是上一步生成的文件
ids1 = [i[:-4] for i in df.slide_id]
ids2 = [i[:-3] for i in os.listdir(r"E:\glptc\toy_test\patches")]
df['slide_id'] = ids1
ids = df['slide_id'].isin(ids2)
sum(ids)
df.loc[ids].to_csv(r"E:\glptc\toy_test\step2.csv",index=False)
