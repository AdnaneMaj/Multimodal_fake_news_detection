import os
import json
import csv
import pandas as pd
from tqdm import tqdm
from typing import Union, List
from ..Enums import BaseEnum
from ..models.yolo import ObjectExtractor

class DatasetCreator:
    def __init__(self, data_dir:str=BaseEnum.DATA_PATH.value,multimodality:bool=False):
        """
        Initialize the DatasetCreator.

        Args:
            parent_dir (str): Path to the parent directory containing all subjects.
            output_csv_path (str): Path to save the output CSV file.
        """
        self.data_dir = data_dir
        self.multimodality = multimodality

        self.parent_dir = os.path.join(data_dir,'raw/PHEME/all-rnr-annotated-threads')
        self.output_csv_path = self.set_output_csv()
        self.class_labels = {"rumours": 0, "non-rumours": 1}
        self.dataset = []
        self.df = None

        if multimodality:
            self.obj_ext = ObjectExtractor()
        
        #Try to get the dataframe if it's already exist
        if not os.path.exists(self.output_csv_path):
            self.process_directories()
            self.save_to_csv()
        self.df = self.get_dataframe()

    def set_output_csv(self):
        """
        Set the name of csv file
        """
        # Start with the base file name
        file_name = "data"

        # Append parts to the file name based on the boolean values
        if self.multimodality:
            file_name += "_multi"

        # Add the file extension
        file_name += ".csv"

        return os.path.join(self.data_dir,'processed',file_name)


    def process_directories(self):
        """
        Process all subject directories within the parent directory.
        """
        subjects = [os.path.join(self.parent_dir, d) for d in os.listdir(self.parent_dir) if os.path.isdir(os.path.join(self.parent_dir, d))]
        for subject_path in subjects:
            subject_name = os.path.basename(subject_path)
            for category, label in self.class_labels.items():
                category_path = os.path.join(subject_path, category)
                if not os.path.exists(category_path):
                    print(f"Directory {category_path} does not exist. Skipping.")
                    continue

                self._process_category(category_path, label, subject_name)

    def _process_category(self, category_path, label, subject):
        """
        Process a specific category directory (e.g., 'rumours' or 'non-rumours').

        Args:
            category_path (str): Path to the category directory.
            label (int): Class label for the category.
            subject (str): Name of the subject (folder name).
        """
        for root, dirs, files in tqdm(os.walk(category_path),desc=f'Downloading subject : {subject}'):
            if os.path.basename(root) == "source-tweets":
                id_post = os.path.basename(os.path.dirname(root))  # Extract ID from parent directory name
                for file in files:
                    if file.endswith(".json") and not file.startswith("._"):
                        file_path = os.path.join(root, file)
                        self._process_file(file_path, id_post, label, subject)

    def _process_file(self, file_path, id_post, label, subject):
        """
        Process a single JSON file and extract relevant data.

        Args:
            file_path (str): Path to the JSON file.
            id_post (str): ID of the post (from the directory name).
            label (int): Class label for the file.
            subject (str): Name of the subject (folder name).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"Skipping invalid file: {file_path}")
            return

        # Extract fields
        text = data.get("text", "")
        media_urls = [
            media.get("media_url", "") for media in data.get("entities", {}).get("media", [])
        ]
        media_urls = ";".join(media_urls)  # Combine multiple URLs into a single string

        #Add detected object as words that occured in the text
        if self.multimodality:
            text += ' '+self.obj_ext.process_single_image(media_url=media_urls)

        # Append to dataset
        self.dataset.append([id_post, label, subject, text, media_urls])

    def save_to_csv(self):
        """
        Save the dataset to a CSV file.
        """
        with open(self.output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            # Write header
            writer.writerow(["id_post", "class", "subject", "text", "Media_url"])
            # Write rows
            writer.writerows(self.dataset)

        print(f"Dataset saved to {self.output_csv_path}")

    #________________________

    def get_dataframe(self):
        """
        Get the dataset as a pandas dataframe
        """
        df = pd.read_csv(self.output_csv_path)
        return df
    
    def get_attributes_by_id(self, id:int, attributes:Union[str,List[str]]):
        """
        Get values of some attributes from the dataframe

        :return :A tuple of desired values
        """
        #Check if attributes is a list or one str, and make it a list if it's a str
        if isinstance(attributes,str):
            attributes = [attributes]

        values = self.df[self.df.id_post == id][attributes].values[0] #This will return a single value if isinstance(attributes,str) true else a tuple

        return values[0] if len(attributes) == 1 else tuple(values)
