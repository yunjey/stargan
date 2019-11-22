# from code import LANDMARK_DATA
import pandas as pd
import os

LANDMARK_DATA = os.path.join('./data','landmarks')

df = pd.read_csv(os.path.join(LANDMARK_DATA,'train.csv'))
df_labels = pd.read_csv(os.path.join(LANDMARK_DATA,'train_label_to_category.csv'))

df = df.merge(df_labels, on='landmark_id', how='left')

# print(df.landmark_id.value_counts()[:40])
df.groupby(['landmark_id', 'category'],as_index=False).count().to_csv(os.path.join(LANDMARK_DATA,'train_counts.csv'))
