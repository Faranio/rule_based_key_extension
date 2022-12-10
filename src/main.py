from PIL import Image
from typing import List

import numpy as np
import pytesseract

from src.text_region_detection import locate_text_regions


def center_point(coords: List[List[float]]) -> List[float]:
    """
    Calculate the coordinates of the central point of a rectangle.
    :param coords: List of coordinates [[x_start, y_start], [x_end, y_end]].
    :return: Coordinates of the central point [x, y].
    """
    return [(coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2]


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert the input image to grayscale image.
    :param image: Input image.
    :return: Grayscale image.
    """
    return np.array(Image.fromarray(image).convert('L'))


def thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply binary and OTSU thresholding to the image.
    :param image: Input image.
    :return: Image with the binary and OTSU thresholding applied on it.
    """
    return binarize_by_otsu(image)


def binarize_by_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    return ((img >= threshold)*255).astype('uint8')


def binarize_by_otsu(img: np.ndarray) -> np.ndarray:
    """Returns a binary version of the image by applying a thresholding operation."""
    otsu_threshold = 0
    lowest_criteria = np.inf
    
    for threshold in range(255):
        thresholded_im = img >= threshold
        # Compute weights
        weight1 = np.sum(thresholded_im) / img.size
        weight0 = 1 - weight1

        # If one the classes is empty, that threshold will not be considered
        if weight1 != 0 and weight0 != 0:
            # Compute criteria, based on variance of these classes
            var0 = np.var(img[thresholded_im == 0])
            var1 = np.var(img[thresholded_im == 1])
            otsu_criteria = weight0 * var0 + weight1 * var1

            if otsu_criteria < lowest_criteria:
                otsu_threshold = threshold
                lowest_criteria = otsu_criteria

    return binarize_by_thresholding(img, otsu_threshold)


def longest_common_substring(S: str, T: str) -> str:
    """
    Find the longest common substring of two strings.
    :param S: Input string 1.
    :param T: Input string 2.
    :return: Longest common substring of two strings.
    """
    m, n = len(S), len(T)
    counter = [[0]*(n+1) for _ in range(m+1)]
    longest = 0
    end_index = m
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if S[i-1] == T[j-1]:
                counter[i][j] = counter[i-1][j-1] + 1
                
                if counter[i][j] > longest:
                    longest = counter[i][j]
                    end_index = i

    return S[end_index-longest:end_index]


def intersects(rectA: List[List[float]], rectB: List[List[float]]) -> bool:
    """
    Find whether two rectangles intersect with each other or not.
    :param rectA: List of coordinates of first rectangle [[x_start, y_start], [x_end, y_end]].
    :param rectB: List of coordinates of second rectangle [[x_start, y_start], [x_end, y_end]].
    :return: Boolean value indicating the intersection.
    """
    if max(rectA[0][0], rectB[0][0]) > min(rectA[1][0], rectB[1][0]) or max(rectA[0][1], rectB[0][1]) > min(rectA[1][1], rectB[1][1]):
        return False
    
    return True


def closest_field_name(doc_image: np.array, general_field_name: str, coords: List[List[float]],
                       possible_fields: List[str], buffer=5) -> str:
    """
    Map the general field name to the specialized field name based on the closest text block in the document image.
    :param doc_image: Target document image represented as numpy array.
    :param general_field_name: A general field name of the detected text block.
    :param coords: Coordinates of the detected field value [[x_start, y_start], [x_end, y_end]].
    :param possible_fields: A list of all possible keys defined by the user.
    :param buffer: A value denoting the number of pixels for increasing the size of the detected text region for
    readability.
    :return: Closest field name from the list of possible fields.
    """
    text_regions = locate_text_regions(doc_image)

    if len(text_regions) == 0:
        print("[INFO] No text regions were detected.")
        return general_field_name
    
    (field_x_start, field_y_start), (field_x_end, field_y_end) = coords
    field_center_x, field_center_y = center_point(coords)
    
    closest_regions = []
    ocr_config = r'--oem 3 --psm 6'
    
    for text_region in text_regions:
        (x_start, y_start), (x_end, y_end) = text_region
        text_center_x, text_center_y = center_point(text_region)
        dist = np.linalg.norm(np.array([text_center_x, text_center_y]) - np.array([field_center_x, field_center_y]))
        
        if not intersects(coords, text_region) and \
                (field_x_start <= x_end and field_x_end >= x_start and field_center_y >= text_center_y
                 or
                 field_y_start <= y_end and field_y_end >= y_start and field_center_x >= text_center_x):
            image_region = doc_image[y_start - buffer:y_end + buffer, x_start - buffer:x_end + buffer]
            image_region = convert_to_grayscale(image_region)
            image_region = thresholding(image_region)
            
            field_term = pytesseract.image_to_string(image_region, config=ocr_config).replace("\n\x0c", '')
            closest_regions.append({"distance": dist, "field_term": field_term, "region": text_region})

    closest_regions = sorted(closest_regions, key=lambda x: x["distance"])
    longest_substring = ''
    result = general_field_name

    for region in closest_regions:
        for field_name in possible_fields:
            substring = longest_common_substring(field_name.strip().lower(), region["field_term"].strip().lower())
            if len(substring) > len(longest_substring):
                longest_substring = substring
                result = field_name
                
    return result
