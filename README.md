# aRTF Clothes
The almost-ready-to-fold (aRTF) Clothes dataset contains around 2000 images of various clothing items across 4 categories, in a variety of realistic (where a human/robot might do laundry folding) household settings.

The almost-ready-to-fold part refers to the states of the clothes. We mimick the output of current SOTA robotic unfolding pipelines, which are capable but not yet perfect.

The aim of this dataset is to evaluate perception modules on a variety of cloth pieces for each category in a number of realistic environments. We hope this will help bring robots out of the labs into our everyday living environments.


TODO: overview table of the dataset statistics (#clothes, #settings, #cameras,#images,...)
TODO: add some images do demonstrate.

## Using this dataset
TODO: provide download links etc.


----
## Making Creation Process

### Data capturing
TODO
### Labeling
CVAT was used for labeling. We set up CVAT locally and also used the serverless components to enable Segment-Anything, which we used for labeling segmentation masks of all cloth items. Keypoints were all manually labeled.

The semantic keypoints we have labeled are illustrated below:
TODO: template + semantic labels for each category.


### Dataset Post-processing

#### Obtaining desired dataset formats
The labeled datasets can be exported from CVAT in the YOLO format (detection or segmentation) and the COCO instances format (detection/segmentation). The COCO keypoints format is atm only supported when skeletons are used. We do not use this but have an alternative flow in which we export the data in the CVAT Images format and then use custom code to convert that into a COCO Keypoints dataset.

To create COCO keypoints datasets, we use the `airo-dataset-tools` package.

1. Export the dataset annotations from cvat in their image format and save it in the parent dir of the dataset images
2. Convert to COCO format using `airo-dataset-tools convert-cvat-to-coco-keypoints --add_segmentation  <path-to-cvat-xml `
3. (if needed) change the relative base of the image paths to match your coco dataset structure.
4. inspect the dataset and annotations with Fiftyone to make sure all looks fine: `airo-dataset-tools fiftyone-coco-viewer <path-to-coco-json> -l keypoints -l segmentations -l detections`



#### Resizing
To facilite rapid iterations, you can resize the images: `airo-dataset-tools resize-coco-keypoints-dataset --width 256 --height 256 <path-to-coco-json>`

#### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`

