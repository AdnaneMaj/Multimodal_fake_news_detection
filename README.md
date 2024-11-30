# Multimodal_fake_news_detection

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

Place the dataset folder after extraction inside [data/raw/](data/raw/) directory with the name **PHEME**

## To-DO

- [ ] Start redacting a paper ( First Draft )
- [x] Create Script to convert an event to csv file
- [ ] Start coding the different part
    - [ ] Graph constructure (textual first)
    - [ ] Word extractor from image using YOLO
    - [ ] Knowledge distilator ( using knowledge graph )
- [ ] First tests

## Ideas/Version

- Try text classification on text only
- Add YOLO/Add knowledge graph

* For the embedding they use simply word embedding, why don't use an encoder ? ( Which will capt more semantic relationship )
    - Here we will have a chauvauchment since the words comming from yolo are not in the context, so why don't use here prompt engineering to incorporate them ( for example : new_text = text + ", this post was accompagend with and image containing the follwing object : "+objects_detected_with_yolo )