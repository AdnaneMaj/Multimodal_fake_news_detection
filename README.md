# Multimodal_fake_news_detection

## First results
![alt text](outputs/20241203_141534_acc_0.8352/20241203_141534_acc_0.8352_training_metrics.png)

## Setup installation

Create the conda environment :
```bash
$ conda env create --file environment.yml
```

Activate the conda env :
```bash
$ conda activate GNN_project
```

Update the conda environment (*if needed*) :
```bash
$ conda env update --file environment.yml
```

## Usage

### 1. Download PHEME data
Donwload the **PHEME** dataset : [PHEME dataset for Rumour Detection and Veracity Classification](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

Place the dataset folder after extraction inside [data/raw/](data/raw/) directory with the name **PHEME**

> You can use a processed version of data from the table below,downlaod the file, extract it and copy it's content to [data/processed](data/processed)

**Processed Data Versions**

| Version | Description                                | Download Link                                                                                          |
|---------|--------------------------------------------|-------------------------------------------------------------------------------------------------------|
| 0       | No multimodality,only bert or w2v        | [Drive Link](https://drive.google.com/drive/folders/19KOAZhz6i5TOTwZVYuAz69WA4iTZ-fzO?usp=sharing)    |
| 1       | Added multimodality <br> The 4 combos of : <br> - w2v<br> - w2v-multi<br> - bert<br> - bert-multi *(Not added yet)*    | [Drive Link](https://drive.google.com/drive/folders/1aE52PN0Mf_2IeOId8Az7Rm8ZAeypy8US?usp=sharing)                   |


### 2. Donwload Google's Word2Vec pretrained model
Donwload the model : [GoogleNews-vectors](https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/blob/main/GoogleNews-vectors-negative300.bin)

Place the `.bin` after extraction inside [src/models](src/models) directory with the name **GoogleNews-vectors-negative300.bin**

## To-DO

- [ ] Start redacting a paper ( First Draft )
- [x] Create Script to convert an event to csv file
- [x] Start coding the different part
    - [x] Graph constructure (textual first)
    - [x] Word extractor from image using YOLO
    - [ ] Knowledge distilator ( using knowledge graph )
- [x] First tests

## Ideas/Version

- Try text classification on text only
- Add YOLO/Add knowledge graph

* For the embedding they use simply word embedding, why don't use an encoder ? ( Which will capt more semantic relationship )
    - Here we will have a chauvauchment since the words comming from yolo are not in the context, so why don't use here prompt engineering to incorporate them ( for example : new_text = text + ", this post was accompagend with and image containing the follwing object : "+objects_detected_with_yolo )

- Use a Graph data base to store the data : https://chatgpt.com/share/674c25eb-6df4-800d-89d4-fa5c3c3cb495


---

## Setting Up the YOLO Model

### ImageAnalyzer Class with YOLO Model Integration

The `ImageAnalyzer` class, located in the **features directory**, leverages the pre-trained YOLO model (`yolo11n.pt`) for object detection. This class allows you to detect various objects in images, extract bounding boxes, and calculate confidence scores for detected objects.

### How to Use the ImageAnalyzer Class

#### Step 1: Download the Model

To use the YOLO model with the `ImageAnalyzer` class, you'll first need to download the pre-trained model (`yolo11n.pt`) and place it in the `models/yolo/` directory.

1. **Manual Download**: 
   - Download the YOLO model file (`yolo11n.pt`) from the following link:
     - [Download yolo11.pt model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) 
   - Once downloaded, move or copy the model file into the `models/yolo/` directory of the project.

2. For further information about the model and other options, please refer to the official [YOLO Detection Models documentation](https://docs.ultralytics.com/tasks/detect/#models).

### Step 2: Use the ImageAnalyzer Class for Object Detection

Once the model is in place, you can use the `ImageAnalyzer` class to perform object detection on images. Here's an example of how to use it:

```python
from features.image_analyzer import ImageAnalyzer

# Initialize the ImageAnalyzer class
image_analyzer = ImageAnalyzer(model_path="models/yolo/yolo11n.pt")

# Run object detection on an image
results = image_analyzer.detect_objects("path/to/your/image.jpg")

# Print the detection results
print(results)
