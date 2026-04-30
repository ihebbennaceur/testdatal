from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import os


class TableDetectionError(Exception):
    """Exception for table detection errors"""
    pass


class TableDetection:
    """Class to detect and extract tables from document images using DETR model"""
    
    def __init__(self, model_name="TahaDouaji/detr-doc-table-detection"):
        """Initialize the table detection model
        
        Args:
            model_name: Model identifier from Hugging Face
        """
        try:
            self.model_name = model_name
            self.pipe = pipeline("object-detection", model=model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        except Exception as e:
            raise TableDetectionError(f"Failed to load model: {str(e)}")

    def extract_tables(self, image_path, confidence_threshold=0.5):
        """Extract tables from an image
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence score (0-1)
            
        Returns:
            List of detected tables
            
        Raises:
            TableDetectionError: If image not found or invalid
        """
        # Validate input
        if not isinstance(image_path, str):
            raise TableDetectionError("Image path must be a string")
            
        if not os.path.exists(image_path):
            raise TableDetectionError(f"Image not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Run detection using pipeline
            results = self.pipe(image)
            
            # Filter for tables with confidence threshold
            tables = [
                result for result in results 
                if result.get('label') == 'table' and result.get('score', 0) >= confidence_threshold
            ]
            
            return tables
        except TableDetectionError:
            raise
        except Exception as e:
            raise TableDetectionError(f"Error extracting tables: {str(e)}")
    

