# CelebA images
URL=https://www.dropbox.com/s/3e5cmqgplchz85o/CelebA_nocrop.zip?dl=0
ZIP_FILE=./data/CelebA_nocrop.zip
mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE

# CelebA attribute labels
URL=https://www.dropbox.com/s/auexdy98c6g7y25/list_attr_celeba.zip?dl=0
ZIP_FILE=./data/list_attr_celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE
