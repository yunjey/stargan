from code import TRANSIENT_DATA, TRANSIENT_DATA_ANNOTATIONS, TRANSIENT_DATA_ATTRIBUTES
from sklearn.model_selection import train_test_split
from shutil import copyfile
import os
import pandas as pd
'''
Read the attribute and annotations files.

Return:
    df (Pandas dataframe): dataframe containing annotations for each image in the dataset.
'''
def get_data():
    with open(TRANSIENT_DATA_ATTRIBUTES, 'r') as f:
        attr = f.read().splitlines()

    df_annotations = pd.read_csv(TRANSIENT_DATA_ANNOTATIONS, sep='\t', header=None, index_col=0, names=attr)

    return df_annotations

'''
Remove the confidence scores (less than 1% of the data has a low confidence score -- so disregarding), and get
the attribute with the maximum probability score.

Arguments:
    df (Pandas dataframe): dataframe containing annotations for each image in the dataset.

Return:
    df (Pandas dataframe): dataframe containing annotations for each image in the dataset with single max 
                            attribute per image.
'''
def preprocess(df):
    #remove confidence scores
    df = df.astype(str)
    df = df.apply(lambda x: x.str.split(',').str[0])
    df = df.astype(float)

    #get attribute with max score
    df['max'] = df.idxmax(axis=1)

    return df

'''
Process the input df and create the proper folder structure.

Arguments:
    df (Pandas dataframe): dataframe containing annotations for each image in the dataset with single max 
                            attribute per image.
'''
def data_struct(df):
    image_paths = list(df.index)
    attr_labels = list(df['max'])

    source_image = TRANSIENT_DATA + 'imageLD' + os.sep
    dest_train_image = TRANSIENT_DATA + 'train' + os.sep
    dest_test_image = TRANSIENT_DATA + 'test' + os.sep

    #create directories for each attribute
    for attr in attr_labels:
        if not os.path.exists(dest_train_image + attr):
            os.makedirs(dest_train_image + attr)

        if not os.path.exists(dest_test_image + attr):
            os.makedirs(dest_test_image + attr)

    #split data into train (80%) and test (20%)
    train, test = train_test_split(range(len(image_paths)), test_size=0.2)

    #copy files in train into right folder with right attribute
    for i in train:
        split_img_path = image_paths[i].split('/')
        dest_image = split_img_path[0] + '_' + split_img_path[1]

        source = source_image + image_paths[i]
        dest = dest_train_image + attr_labels[i] + os.sep + dest_image

        copyfile(source, dest)

    #copy files in test into right folder with right attribute
    for i in test:
        split_img_path = image_paths[i].split('/')
        dest_image = split_img_path[0] + '_' + split_img_path[1]

        source = source_image + image_paths[i]
        dest = dest_test_image + attr_labels[i] + os.sep + dest_image

        copyfile(source, dest)

def main():
  df_annotations = preprocess(get_data())
  print(df_annotations)
  data_struct(df_annotations)
  
if __name__== "__main__":
  main()
