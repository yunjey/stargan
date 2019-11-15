from code import LANDMARK_DATA
import pandas as pd
import os
import urllib
from sklearn.model_selection import train_test_split


'''
Get the landmark name corresponding to the landmark id

Return:
	dictionary : containing mapping of landmark name to landmark id
'''

def get_landmark_ids_with_name(ids):
	landmarks_ids = {}

	id_to_name_df = pd.read_csv(os.path.join(LANDMARK_DATA,'train_label_to_category.csv'))
	for l_id in ids:
		landmark_url = id_to_name_df[(id_to_name_df.landmark_id==l_id)].category
		for index,value in landmark_url.items():
			landmark = value.split(':')[2]
			landmarks_ids[landmark] = l_id

	return landmarks_ids

'''
Download images using the urls given in the train.csv file

Return: 
	void
'''
def download_images():
	#reading the csv having the landmark_id to landmark image url mapping
	df = pd.read_csv(os.path.join(LANDMARK_DATA, 'train.csv'))

	#getting the 20 landmark ids which have the maximum number of images
	df1 = df.groupby(['landmark_id'], sort=False).size()
	df1 = df1.sort_values(ascending=False)

	df1 = df1[1:21] # not selecting index 0 since it is not for a particular landmark
	ids_list = []
	for index,value in df1.items():
		ids_list.append(index)

	landmarks_ids = get_landmark_ids_with_name(ids_list)
	
	#create test and train directories
	train_dir = os.path.join(LANDMARK_DATA ,'train')
	test_dir = os.path.join(LANDMARK_DATA, 'test')

	if not os.path.exists(train_dir):
		os.mkdir(train_dir)

	if not os.path.exists(test_dir):
		os.mkdir(test_dir)

	for landmark in landmarks_ids:
			
			print(landmark)
			landmark_id = landmarks_ids[landmark]

			#selecting 400 images for each landmark
			landmark_df = df[(df.landmark_id==landmark_id)][0:400]
			landmark_train_dir = os.path.join(train_dir, landmark)
			landmark_test_dir = os.path.join(test_dir, landmark)

			if not os.path.exists(landmark_train_dir):
				os.mkdir(landmark_train_dir)

			if not os.path.exists(landmark_test_dir):
				os.mkdir(landmark_test_dir)

			count = 0
			urls = list(landmark_df.url)
			img_ids = list(landmark_df.id)

			#split image ids into 90% train and 10% test
			train,test = train_test_split(range(len(img_ids)), test_size=0.1)

			for i in train:
				print('Downloading image : '+ str(i))
				image_path = landmark_train_dir + os.sep + img_ids[i] +'.jpg'
				try:
					#downloading the image
					urllib.request.urlretrieve(urls[i], image_path)
				except Exception as e:
					print(e)
					continue

			for i in test:
				print('Downloading image : '+ str(i))
				image_path = landmark_test_dir + os.sep + img_ids[i] +'.jpg'
				try:
					urllib.request.urlretrieve(urls[i], image_path)
				except Exception as e:
					print(e)
					continue


if __name__ == '__main__':

	if not os.path.exists(LANDMARK_DATA):
		os.mkdir(LANDMARK_DATA)
	download_images()