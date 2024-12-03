import pandas as pd
import logging
from tqdm import tqdm
from image_analyzer import ImageAnalyzer

class SimpleImageFeatureExtractor:
    """
    A class to extract basic image features (detected objects and confidence)
    from a dataset using ImageAnalyzer
    """
    
    def __init__(self, image_analyzer: ImageAnalyzer):
        """
        Initialize the feature extractor with an ImageAnalyzer instance
        
        Args:
            image_analyzer: An initialized ImageAnalyzer object
        """
        self.analyzer = image_analyzer
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_single_image(self, media_url: str) -> dict:
        """
        Process a single image and extract basic features
        
        Args:
            media_url: URL of the image to analyze
            
        Returns:
            Dict containing detected_objects and image_confidence
        """
        try:
            results = self.analyzer.analyze_image_from_url(
                media_url, 
                save_annotated=False
            )
            
            # Extract only required features
            features = {
                'detected_objects': '|'.join(results['objects']),
                'image_confidence': float(results['image_confidence'])
            }
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to process image {media_url}: {e}")
            return None
    
    def process_dataset(self, 
                       df: pd.DataFrame, 
                       media_url_col: str = 'Media_url',
                       batch_size: int = 100) -> pd.DataFrame:
        """
        Process all images in the dataset and add feature columns
        
        Args:
            df: Input DataFrame
            media_url_col: Name of the column containing image URLs
            batch_size: Number of images to process in each batch
            
        Returns:
            DataFrame with added image feature columns
        """
        # Create empty columns for features
        df['detected_objects'] = None
        df['image_confidence'] = None
            
        # Process images in batches with progress bar
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                if pd.notna(row[media_url_col]):
                    features = self.process_single_image(row[media_url_col])
                    
                    if features:
                        df.at[idx, 'detected_objects'] = features['detected_objects']
                        df.at[idx, 'image_confidence'] = features['image_confidence']
                            
        return df

# def main():
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     try:
#         # Load the dataset
#         df = pd.read_csv('/home/anas-nouri/OrganizeData/AllDataset1.csv')
        
#         # Initialize ImageAnalyzer and SimpleImageFeatureExtractor
#         analyzer = ImageAnalyzer()
#         extractor = SimpleImageFeatureExtractor(analyzer)
        
#         # Process the dataset
#         logger.info("Starting image feature extraction...")
#         df_with_features = extractor.process_dataset(df)
        
#         # Save the enhanced dataset
#         output_path = 'dataset_with_image_features.csv'
#         df_with_features.to_csv(output_path, index=False)
#         logger.info(f"Enhanced dataset saved to {output_path}")
        
#         # Print summary statistics
#         logger.info("\nFeature extraction summary:")
#         logger.info(f"Total images processed: {len(df)}")
#         logger.info(f"Images with detected objects: {df_with_features['detected_objects'].notna().sum()}")
#         logger.info(f"Average confidence score: {df_with_features['image_confidence'].mean():.2f}")
        
#     except Exception as e:
#         logger.error(f"Feature extraction failed: {e}")
#         raise

# if __name__ == "__main__":
#     main()