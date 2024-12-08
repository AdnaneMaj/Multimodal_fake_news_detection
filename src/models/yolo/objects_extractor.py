import cv2
import numpy as np
import requests
from typing import List, Dict
from ultralytics import YOLO
from ...Enums import BaseEnum

class ObjectExtractor:
    """
    A comprehensive image analysis class that performs object detection 
    """

    def __init__(self, model_path: str = BaseEnum.YOLO_MODEL.value, confidence_threshold: float = 0.25):
        """
        Initialize the ObjectExtractor with a YOLO model and configuration.

        Args:
            model_path (str): Path to the YOLO model weights file.
            confidence_threshold (float): Minimum confidence score for detections.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def download_image(self, image_url: str) -> np.ndarray:
        """
        Download an image from a given URL.

        Args:
            image_url (str): URL of the image to download.

        Returns:
            np.ndarray: Decoded image array.

        Raises:
            ValueError: If image cannot be downloaded or decoded.
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode the image.")
            
            return image
        except requests.RequestException as e:
            return None

    def detect_objects(self, image: np.ndarray) -> List[str]:
        """
        Detect and analyze objects in the image using YOLO.

        Args:
            image (np.ndarray): Input image for object detection.

        Returns:
            List containing detected objects names.
        """
        if image is None:
            return []
        try:
            # Run YOLOv8 detection
            results = self.model.predict(
                source=image, 
                conf=self.confidence_threshold, 
                save=False, 
                save_txt=False,
                verbose=False
            )

            # Extract detection information
            labels = results[0].names
            detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]

            return detected_classes
        except Exception as e:
            return []

    def analyze_image_from_url(self, image_url: str) -> Dict:
        """
        Comprehensive image analysis from a URL.

        Args:
            image_url (str): URL of the image to analyze.

        Returns:
            Dict containing analysis results: objects, OCR results, and image path.
        """
        # Download image
        image = self.download_image(image_url)

        # Detect objects
        detected_classes = self.detect_objects(image)

        return detected_classes

        
    def process_single_image(self, media_url: str) -> str:
        """
        Process a single image and extract basic features
        
        Args:
            media_url: URL of the image to analyze
            
        Returns:
            Dict containing detected_objects and image_confidence
        """
        detected_classes = self.analyze_image_from_url(media_url)
        detected_classes_string = " ".join(element.replace(" ", "-") for element in detected_classes)

        return detected_classes_string
