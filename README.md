# iMaterialist (Fashion) 2019 Competition solution developed by "Гиперболические ИПСоиды" team for the INT20TH Hackathon selection

## Model selection
#### Among many models available today for instance segmentation we've chosen simple **Mask R-CNN** with a Resnet 50 backbone.

Analysing sources like https://paperswithcode.com/task/instance-segmentation and looking into specifics of each State of The Art (STA) models we've concluded:
-  *Mask R-CNN* is the easiest and quickest one to train
- despite its age, it's still considered as on of the STA approaches with one of the highest scores img1, img2
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
## Results

## Problems we've encountered
A very samll amount of time given to solve this task. Taking into account a massive dataset and gigantic models required to achive any acceptable results, working locally wasn't an option. Other free platforms like Google Colab or Kaggle Kernels had multiple restrictions and their efficency still wasn't enough to try all the models and approaches.
So time and resource limitations were our main problems.
## Ways to improve
- do more epochs
- tune training parameters to get a better model
-  use the fact that the position of the clothing item is related to a specific body part (use another model to detect body parts)
- try other backbones like Feature Pyramid Network (FPN) or a ResNet101
- test other models and architectures (like *Hybrid Task Cascade with ResNeXt-101-64x4d-FPN backbone*)
- ensemble best different models
