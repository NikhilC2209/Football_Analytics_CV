# Football Analytics using Yolov8

<a href="https://universe.roboflow.com/nikhil-chapre-xgndf/detect-players-dgxz0">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

<a href="https://universe.roboflow.com/nikhil-chapre-xgndf/detect-players-dgxz0/model/">
    <img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>


<p></p>

![Sample GIF](https://github.com/NikhilC2209/Football_Analytics_CV/blob/master/sample/final.gif)


## Data Sources:

- DFL — Bundesliga Data Shootout from Kaggle, this data source contains clips from Bundesliga matches provided publicly by the Deutsche Fußball Liga (DFL). Data contains both short clips and long full match recordings. Link: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data
- SoccerNet: SoccerNet is a large-scale dataset for soccer video understanding, it consists of a large array of tasks based on video data taken from major European leagues. SoccerNet also has its own python package with rich documentation for ease of use. Link: https://www.soccer-net.org/data
- Scraping data through Youtube highlights of major leagues from their official channels by directly importing them through Roboflow or using cli tools to download them.
- Using self-recorded clips of more recent matches from the Top 5 leagues for clearer quality.
- Using clips from games such as Pro Evolution Soccer and FIFA to add diversity to our dataset and make our model more robust. Another advantage here is that we can select the camera angle of our liking to get better, less clustered images.

## SETUP

```
python -m venv YOLO_CV
YOLO_CV\Scripts\activate

pip install -r requirements.txt
```

Download hand-annotated Dataset from Roboflow Universe: https://universe.roboflow.com/nikhil-chapre-xgndf/detect-players-dgxz0

## TRAINING ON CUSTOM DATASET

Prepare a config.yaml file 

```
path: D:\Github\Football_Analytics_CV\dataset                # absolute path to your dataset folder
train: train                                                 
test: test                                                   # relative path to your train/test/val folders
val: valid

# Label all classes in your dataset

names:
  0: Ball
  1: Player
  2: Referee
```

Then Run ```python train.py``` with any yolov8 weight of your choice (preferrably yolovn.pt/yolov8m.pt) 

Then access the weights from ```runs/detect/predict/weights``` to run inference

## INFERENCE

To run inference for simple player and ball tracking run ```python detect_player.py``` by specifying weight from the trained model. 

The inference video is stored at ```runs/detect/predict``` in .avi format
