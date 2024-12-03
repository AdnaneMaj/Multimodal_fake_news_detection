import cv2
import numpy as np
import requests
from typing import List, Dict, Union
from ultralytics import YOLO
from pytesseract import image_to_string
import logging

class ImageAnalyzer:
    """
    A comprehensive image analysis class that performs object detection 
    and optical character recognition (OCR) on images from URLs.
    """

    def __init__(self, 
                 model_path: str = "models/yolo/yolo11n.pt", 
                 confidence_threshold: float = 0.25, 
                 language: str = "eng"):
        """
        Initialize the ImageAnalyzer with a YOLO model and configuration.

        Args:
            model_path (str): Path to the YOLO model weights file.
            confidence_threshold (float): Minimum confidence score for detections.
            language (str): Language for OCR text extraction.
        """
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            self.ocr_language = language
            self.logger = logging.getLogger(self.__class__.__name__)
            logging.basicConfig(level=logging.INFO, 
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        except Exception as e:
            self.logger.error(f"Failed to initialize ImageAnalyzer: {e}")
            raise

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
            self.logger.error(f"Image download failed: {e}")
            raise ValueError(f"Failed to download image: {e}")

    def detect_objects(self, image: np.ndarray) -> Dict[str, Union[List[str], float]]:
        """
        Detect and analyze objects in the image using YOLO.

        Args:
            image (np.ndarray): Input image for object detection.

        Returns:
            Dict containing detected objects, their counts, and image confidence.
        """
        try:
            # Run YOLOv8 detection
            results = self.model.predict(
                source=image, 
                conf=self.confidence_threshold, 
                save=False, 
                save_txt=False
            )

            # Extract detection information
            labels = results[0].names
            confidences = results[0].boxes.conf
            detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]

            # Count detected classes
            class_counts = {label: detected_classes.count(label) for label in set(detected_classes)}

            # Calculate overall confidence
            total_confidence = sum(confidences)
            total_count = len(confidences)
            average_confidence = total_confidence / total_count if total_count > 0 else 0

            # Prepare detection summary
            detected_objects = [f"{count} {label}" for label, count in class_counts.items()]

            return {
                "objects": detected_objects,
                "class_counts": class_counts,
                "image_confidence": average_confidence
            }
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return {"objects": [], "class_counts": {}, "image_confidence": 0}

    def perform_ocr(self, 
                    image: np.ndarray, 
                    detections: List[np.ndarray], 
                    confidences: np.ndarray) -> List[Dict[str, Union[List[int], str, float]]]:
        """
        Perform Optical Character Recognition (OCR) on detected regions.

        Args:
            image (np.ndarray): Original image.
            detections (List[np.ndarray]): Bounding box coordinates.
            confidences (np.ndarray): Confidence scores for detections.

        Returns:
            List of OCR results with bounding box, extracted text, and confidence.
        """
        ocr_results = []
        
        for box, conf in zip(detections, confidences):
            try:
                # Convert bounding box coordinates to integers
                x_min, y_min, x_max, y_max = map(int, box[:4])

                # Crop the detected text region
                cropped_region = image[y_min:y_max, x_min:x_max]

                # Perform OCR
                text = image_to_string(cropped_region, lang=self.ocr_language).strip()

                # Store OCR result
                ocr_results.append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "text": text,
                    "confidence": float(conf)
                })
            except Exception as e:
                self.logger.warning(f"OCR failed for a region: {e}")

        return ocr_results

    def annotate_image(self, 
                       image: np.ndarray, 
                       detections: List[np.ndarray], 
                       confidences: np.ndarray, 
                       ocr_results: List[Dict]) -> np.ndarray:
        """
        Annotate the image with bounding boxes and recognized text.

        Args:
            image (np.ndarray): Original image.
            detections (List[np.ndarray]): Bounding box coordinates.
            confidences (np.ndarray): Confidence scores.
            ocr_results (List[Dict]): OCR results for each detection.

        Returns:
            np.ndarray: Annotated image.
        """
        annotated_image = image.copy()
        
        for box, conf, ocr_result in zip(detections, confidences, ocr_results):
            x_min, y_min, x_max, y_max = map(int, box[:4])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add text label
            label = f"{ocr_result.get('text', '')} ({conf:.2f})"
            cv2.putText(annotated_image, label, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated_image

    def analyze_image_from_url(self, image_url: str, save_annotated: bool = True) -> Dict:
        """
        Comprehensive image analysis from a URL.

        Args:
            image_url (str): URL of the image to analyze.
            save_annotated (bool): Whether to save the annotated image.

        Returns:
            Dict containing analysis results: objects, OCR results, and image path.
        """
        try:
            # Download image
            image = self.download_image(image_url)

            # Detect objects
            detection_results = self.detect_objects(image)

            # Run YOLO detection again to get full results
            yolo_results = self.model.predict(
                source=image, 
                conf=self.confidence_threshold, 
                save=False, 
                save_txt=False
            )

            # Extract detections and confidences
            detections = yolo_results[0].boxes.xyxy
            confidences = yolo_results[0].boxes.conf

            # Perform OCR
            ocr_results = self.perform_ocr(image, detections, confidences)

            # Annotate image
            annotated_image = self.annotate_image(image, detections, confidences, ocr_results)

            # Save annotated image if requested
            image_path = None
            if save_annotated:
                image_path = "annotated_image.jpg"
                cv2.imwrite(image_path, annotated_image)
                self.logger.info(f"Annotated image saved to {image_path}")

            # Prepare and return comprehensive results
            return {
                "objects": detection_results['objects'],
                "class_counts": detection_results['class_counts'],
                "image_confidence": detection_results['image_confidence'],
                "ocr_results": ocr_results,
                "annotated_image_path": image_path
            }

        except Exception as e:
            self.logger.error(f"Complete image analysis failed: {e}")
            raise

# # Example usage
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)

#     # Initialize the analyzer
#     analyzer = ImageAnalyzer()

#     # Analyze an image from URL
#     image_url = "https://pbs.twimg.com/media/B66Ooo1CIAAzODr.jpg"
#     try:
#         analysis_results = analyzer.analyze_image_from_url(image_url)
        
#         # Print analysis results
#         print("Detected Objects:", analysis_results['objects'])
#         print("Class Counts:", analysis_results['class_counts'])
#         print("Image Confidence:", analysis_results['image_confidence'].item())
        
#         # Print OCR results
#         print("\nOCR Results:")
#         for result in analysis_results['ocr_results']:
#             print(f"Text: {result['text']}, Confidence: {result['confidence']:.2f}")
    
#     except Exception as e:
#         print(f"Analysis failed: {e}")