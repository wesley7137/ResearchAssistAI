import cv2
import pytesseract
import os
import openai

# Phase 4: Image and Multimedia Analysis
def preprocess_image(image_path):
    """ Preprocesses an image by converting to grayscale, resizing, and applying adaptive thresholding.

    :param image_path: Path to the image file.
    :return: Preprocessed image.
    """
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size for consistency
    resized = cv2.resize(gray, (800, 800))

    # Apply adaptive thresholding to enhance text regions
    processed_image = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return processed_image

def extract_data_from_image(image):
    """ Extracts text data from a processed image using OCR.

    :param image: Preprocessed image.
    :return: Extracted text data.
    """
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(image)

    # Further processing can be done to extract quantitative data from text
    return text

