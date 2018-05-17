Image classififier, using keypoints and bag of words technique.

- 1) Extract image descriptors using SIFT/SURF
- 2) Perform k-means clustering on the descriptors to generate the dictionary
- 3) Generate histograms of images based on dictionary
- 4) Train SVM using histograms


Train:
```
python findFeatures.py -t dataset/train/
```

Test:
```
python getClass.py -t dataset/test --visualize
```
Result of testing images:

![image_1](https://i.imgur.com/CT9f8qN.png) ![image_2](https://i.imgur.com/aOQ8XPj.png)
 
Result from telegram bot:               

![image_3](https://i.imgur.com/caTapRk.jpg)
