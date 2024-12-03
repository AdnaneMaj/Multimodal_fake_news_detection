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

### 1. Download data
Donwload the **PHEME** dataset : [PHEME dataset for Rumour Detection and Veracity Classification](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

### 2. Donwload Google's Word2Vec pretrained model
Donwload the model : [GoogleNews-vectors](https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/blob/main/GoogleNews-vectors-negative300.bin)


Place the dataset folder after extraction inside [data/raw/](data/raw/) directory with the name **PHEME**

## To-DO

- [ ] Start redacting a paper ( First Draft )
- [x] Create Script to convert an event to csv file
- [x] Start coding the different part
    - [x] Graph constructure (textual first)
    - [x] Word extractor from image using YOLO
    - [ ] Knowledge distilator ( using knowledge graph )
- [ ] First tests

## Ideas/Version

- Try text classification on text only
- Add YOLO/Add knowledge graph

* For the embedding they use simply word embedding, why don't use an encoder ? ( Which will capt more semantic relationship )
    - Here we will have a chauvauchment since the words comming from yolo are not in the context, so why don't use here prompt engineering to incorporate them ( for example : new_text = text + ", this post was accompagend with and image containing the follwing object : "+objects_detected_with_yolo )

- Use a Graph data base to store the data : https://chatgpt.com/share/674c25eb-6df4-800d-89d4-fa5c3c3cb495

## Setting Up the YOLO Model

The YOLO model file (`yolo11.pt`) is required for object detection. It should be placed in the `models/yolo/` directory.

### Step 1: Download the Model

You can manually download the `yolo11.pt` model file and place it in the `models/yolo/` directory.

- **Manual Download**: 
  - Download the YOLO model file (`yolo11.pt`) from the following link:
    - [Download yolo11.pt model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) 
  - Move or copy the model file into the `models/yolo/` directory.
  - for more informations you can see this url (https://docs.ultralytics.com/tasks/detect/#models)