# aRTF Clothes
The aRTF (almost-ready-to-fold) Clothes dataset contains around 2000 images of various clothing items across 4 categories, in a variety of realistic (where a human/robot might do laundry folding) household settings.

The almost-ready-to-fold part refers to the states of the clothes. We mimick the output of current SOTA robotic unfolding pipelines, which are capable but not yet perfect.

The aim of this dataset is to evaluate perception modules on a variety of cloth pieces for each category in a number of realistic environments. We hope this will help bring robots out of the labs into our everyday living environments.



TODO: add some images do demonstrate.


Category|Split| # Scenes| # Cloth items| #images
---|---|---|---|---
Towel | train | 6 | 15 | 210
Towel | test | 8 | 20 | 400
Tshirt | train | 6 | 15 | 210
Tshirt | test | 8 | 20 | 400
Shorts | train | 6 | 8 | 112
Shorts | test | 8 | 9 | 180
Boxershorts | train | 6 |  11 | 154
Boxershorts | test | 8 |11 | 220
Total | train | 6 | 49 | 686
Total | test | 8 | 60 | 1200






## Using this dataset
TODO: provide download links etc.


----
## Dataset Creation

### Data capturing
TODO
### Labeling
CVAT was used for labeling. We set up CVAT locally and also used the serverless components to enable Segment-Anything, which we used for labeling segmentation masks of all cloth items. Keypoints were all manually labeled.

The semantic keypoints we have labeled are illustrated below:
TODO: template + semantic labels for each category.


### Dataset Post-processing
Make sure to add the dataset to the `data` folder in this repo.

#### Local installation (required to run the postprocessing steps)

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


#### Obtaining desired dataset formats
The labeled datasets can be exported from CVAT in the YOLO format (detection or segmentation) and the COCO instances format (detection/segmentation). The COCO keypoints format is atm only supported when skeletons are used. We do not use this but have an alternative flow in which we export the data in the CVAT Images format and then use custom code to convert that into a COCO Keypoints dataset.

To create COCO keypoints datasets, we use the `airo-dataset-tools` package. A number of steps are bundled in [this](artf_clothes/scripts/create_coco_dataset.py) script:

1. Export the dataset annotations from cvat in their image format and save it in the parent dir of the dataset images
2. Convert to COCO format using `airo-dataset-tools convert-cvat-to-coco-keypoints --add_segmentation  <path-to-cvat-xml `
3. (if needed) change the relative base of the image paths to match your coco dataset structure.


4. inspect the dataset and annotations with Fiftyone to make sure all looks fine: `airo-dataset-tools fiftyone-coco-viewer <path-to-coco-json> -l keypoints -l segmentations -l detections`


5. (TODO) convert to YOLO formats.


#### Resizing
To facilite rapid iterations, you can resize the images: `airo-dataset-tools resize-coco-keypoints-dataset --width 256 --height 256 <path-to-coco-json>`


