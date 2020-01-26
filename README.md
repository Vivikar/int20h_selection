# iMaterialist (Fashion) 2019 Competition solution developed by "Гиперболические ИПСоиды" team for the INT20TH Hackathon selection
The jupyter notebook version can be viewed [here ](https://github.com/Vivikar/int20th_selection/blob/master/iMaterialist_INT20TH.ipynb) and the html version [here](https://github.com/Vivikar/int20th_selection/blob/master/iMaterialist_INT20TH.html)

## Model selection
#### Among many models available today for instance segmentation we've chosen simple **Mask R-CNN** with a Resnet 50 backbone.

Analysing sources like https://paperswithcode.com/task/instance-segmentation and looking into specifics of each State of The Art (STA) models we've concluded:
-  *Mask R-CNN* is the easiest and quickest one to train
- despite its age, it's still considered as on of the STA approaches with one of the highest scores
![COCO minival](https://github.com/Vivikar/int20th_selection/blob/master/coco_minival.png)
![COCO test-dev](https://github.com/Vivikar/int20th_selection/blob/master/coco_test_dev.png)
- it fits into kaggle kernels and its training and validating takes ~~a bit less than eternity~~ a relatively small amount of time, so we can fit into the task deadline
- we've had previous experience working with it

## Training pipeline
We've trained our model for 10 epochs, starting from the weights received after training on COCO dataset. Transfer Learning helped us massively in this case since thanks to it 10 epochs were enough to get more or less acceptable results.

To enrich our data and make the model more robust we've augmented our data, using following augmentations:
```
Fliplr(0.5), # horizontal flip
Multiply((0.3, 1.2), per_channel=0.5), # change brightness of images (30-120% of original value)
ContrastNormalization((
Crop(percent=(0, 0.1)), # random crops
arithmetic.JpegCompression((80, 90))
```
For the first 7 epochs, we've trained all the layers of our model, and for the last 3, we've trained only the top part. This allowed us to use as much information as we could from weights from COCO dataset meanwhile paying more attention to the top part of the model, which actually responds for ROI detection and creating masks. However, we thought that training at least for a bit the whole model will help us to improve the quality of the CNN responsible for feature detection to make it better suited for this data.
## Results
Despite training for only 10 epochs and relatively simple pipeline we've managed to achieve a result of **0.0082**. It's the result of the model trained after 10th epoch and tested on 20% of the training data, which were splitted from the begining and the model has never seen before. Since our train AND validation loss kept decreasing and were the lowest on 10th epoch we've chosen to use it as our final checkpoint. However, we are quite sure that if we were to continue training, we would achieve much better result since the model was still underfitting a bit. 
## Problems we've encountered
A very samll amount of time given to solve this task. Taking into account a massive dataset and gigantic models required to achive any acceptable results, working locally wasn't an option.Free platforms like Google Colab or Kaggle Kernels had multiple restrictions and their efficency still wasn't enough to try all the models and approaches.
So time and resource limitations were our main problems.
## Ways to improve
- do more epochs
- tune training parameters 
-  use the fact that the position of the clothing item is related to a specific body part (use another model to detect body parts)
- try other backbones like Feature Pyramid Network (FPN) or a ResNet101
- test other models and architectures (like *Hybrid Task Cascade with ResNeXt-101-64x4d-FPN backbone*)
- ensemble best different models AND (OR) their predictions
