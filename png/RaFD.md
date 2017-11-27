# RaFD dataset guideline
#### 1) Split the dataset into training and test sets (90\%/10\% for training and test, respectively).  
#### 2) Crop all images to 256 x 256, where the faces are centered, and save them in the format shown below.  

    data
    └── RaFD
        ├── train
        |   ├── angry
        |   |   ├── aaa.jpg  (name doesn't matter)
        |   |   ├── bbb.jpg
        |   |   └── ...
        |   ├── happy
        |   |   ├── ccc.jpg
        |   |   ├── ddd.jpg
        |   |   └── ...
        |   ...
        |
        └── test
            ├── angry
            |   ├── eee.jpg
            |   ├── fff.jpg
            |   └── ...
            ├── happy
            |   ├── ggg.jpg
            |   ├── iii.jpg
            |   └── ...
            ...


