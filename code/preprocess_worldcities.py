import pandas as pd
import os
import urllib
from sklearn.model_selection import train_test_split

WORLD_CITIES_DATA = os.path.join('./data','world_cities')


'''
Download images using the urls given in the image_urls.txt file
Return: 
	void
'''
def download_images():
	
	#create test and train directories
	train_dir = os.path.join(WORLD_CITIES_DATA ,'train')
	test_dir = os.path.join(WORLD_CITIES_DATA, 'test')

	if not os.path.exists(train_dir):
		os.mkdir(train_dir)

	if not os.path.exists(test_dir):
		os.mkdir(test_dir)

	# The image_url.txt files has urls and city names separated by a space
	df = pd.read_csv(os.path.join(WORLD_CITIES_DATA, 'image_urls.txt'), sep=" ", header=None)
	df.columns = ['url', 'city']

	#Getting all the city names
	cities_list = df.city.unique()
	number_of_images = {}

	for city in cities_list:
		print(city)
		count = 0

		wc_df = df[(df.city==city)]
		wc_train_dir = os.path.join(train_dir, city)
		wc_test_dir = os.path.join(test_dir, city)

		if not os.path.exists(wc_train_dir):
				os.mkdir(wc_train_dir)

		if not os.path.exists(wc_test_dir):
				os.mkdir(wc_test_dir)

		urls = list(wc_df.url)
		cities = list(wc_df.city)

		#split image ids into 80% train and 20% test
		train,test = train_test_split(range(len(urls)), test_size=0.2)

		for i in train:
			image_name = urls[i].strip().split("/")[-1]
			image_url = urls[i].split(" ")[0]
			image_path = wc_train_dir + os.sep + image_name
			try:
				#downloading the image
				urllib.request.urlretrieve(image_url, image_path)
				count += 1
			except Exception as e:
				print(e)
				continue

		for i in test:
			image_name = urls[i].strip().split("/")[-1]
			image_url = urls[i].split(" ")[0]
			image_path = wc_test_dir + os.sep + image_name
			try:
				urllib.request.urlretrieve(image_url, image_path)
				count += 1
			except Exception as e:
				print(e)
				continue
		number_of_images[city] = count
	print(number_of_images)

if __name__ == '__main__':

	if not os.path.exists(WORLD_CITIES_DATA):
		os.mkdir(WORLD_CITIES_DATA)
	download_images()