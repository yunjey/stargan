## RaFD Dataset Guideline

#### 1. Split images into training and test sets (e.g., 90\%/10\% for training and test, respectively).  
#### 2. Crop all images to 256 x 256, where the faces are centered.
#### 3. Save images in the format shown below:


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


