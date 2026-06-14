import unittest
from unittest.mock import patch
from src.image_processor import process_image
from src.classifier import classify_animal
from src.classifier import _extract_best_breed_from_predictions
from src.classifier import classify_breed
import os

class TestAnimalClassification(unittest.TestCase):
    
    def test_process_image(self):
        # Create a dummy image for testing
        import cv2
        import numpy as np
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        # Draw a rectangle to simulate animal
        cv2.rectangle(dummy_image, (10, 10), (190, 90), (255, 255, 255), -1)
        cv2.imwrite('data/sample.jpg', dummy_image)
        
        measurements = process_image('data/sample.jpg')
        self.assertIsInstance(measurements, dict)
        self.assertIn('body_length', measurements)
        
        # Clean up
        os.remove('data/sample.jpg')
    
    def test_classify_animal(self):
        measurements = {
            'body_length': 120,
            'height_at_withers': 90,
            'chest_width': 60,
            'rump_angle': 40
        }
        score = classify_animal(measurements)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)

    def test_breed_extraction_ignores_non_animal_labels(self):
        predictions = [
            {'label': 'cowboy hat', 'score': 0.98},
            {'label': 'football helmet', 'score': 0.75}
        ]

        animal_type, breed, confidence = _extract_best_breed_from_predictions(predictions)

        self.assertEqual(animal_type, 'Animal')
        self.assertEqual(breed, 'Unknown Breed')
        self.assertEqual(confidence, 0.0)

    def test_breed_extraction_selects_real_breed(self):
        predictions = [
            {'label': 'cowboy hat', 'score': 0.97},
            {'label': 'jersey cow', 'score': 0.68}
        ]

        animal_type, breed, confidence = _extract_best_breed_from_predictions(predictions)

        self.assertEqual(animal_type, 'Cattle')
        self.assertIn('Jersey', breed)
        self.assertGreater(confidence, 0.6)

    @patch('src.classifier.query_huggingface_api')
    def test_classify_breed_rejects_non_livestock_image(self, mock_query_hf):
        mock_query_hf.return_value = [
            {'label': 'sports car', 'score': 0.91},
            {'label': 'tractor', 'score': 0.05}
        ]

        result = classify_breed({'body_length': 250}, 'fake.jpg')
        self.assertIn('No cattle or buffalo detected', result)

    @patch('src.classifier.query_huggingface_api')
    def test_classify_breed_requires_species_verification(self, mock_query_hf):
        mock_query_hf.return_value = None

        result = classify_breed({'body_length': 250}, 'fake.jpg')
        self.assertIn('AI service unavailable', result)

    @patch('src.classifier.query_huggingface_api')
    def test_classify_breed_offline_estimate_when_ai_unavailable(self, mock_query_hf):
        mock_query_hf.return_value = None
        measurements = {
            'body_length': 900,
            'height_at_withers': 500,
            'chest_width': 450,
            'rump_angle': 40,
            'aspect_ratio': 1.6,
            'compactness': 2000,
            'avg_brightness': 145
        }

        result = classify_breed(measurements, 'fake.jpg')
        self.assertIn('Estimated offline', result)

if __name__ == '__main__':
    unittest.main()