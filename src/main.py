from typing import List

import cv2
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
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image: np.ndarray) -> np.ndarray:
	"""
	Apply binary and OTSU thresholding to the image.
	:param image: Input image.
	:return: Image with the binary and OTSU thresholding applied on it.
	"""
	return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


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
                       possible_fields: List[str], buffer=5, show=False) -> str:
	"""
	Map the general field name to the specialized field name based on the closest text block in the document image.
	:param doc_image: Target document image represented as numpy array.
	:param general_field_name: A general field name of the detected text block.
	:param coords: Coordinates of the detected field value [[x_start, y_start], [x_end, y_end]].
	:param possible_fields: A list of all possible keys defined by the user.
	:param buffer: A value denoting the number of pixels for increasing the size of the detected text region for
	readability.
	:param show: Boolean parameter indicating whether to show intermediate results or not.
	:return: Closest field name from the list of possible fields.
	"""
	text_regions = locate_text_regions(doc_image, show=show)
	
	if len(text_regions) == 0:
		print("[INFO] No text regions were detected.")
		return general_field_name
	
	(field_x_start, field_y_start), (field_x_end, field_y_end) = coords
	field_center_x, field_center_y = center_point(coords)
	
	closest_regions = []
	ocr_config = r'--oem 3 --psm 6'
	
	if show:
		doc_image = cv2.rectangle(doc_image, (field_x_start, field_y_start), (field_x_end, field_y_end), (0, 0, 255), 2)

	for text_region in text_regions:
		(x_start, y_start), (x_end, y_end) = text_region
		text_center_x, text_center_y = center_point(text_region)
		dist = np.linalg.norm(np.array([text_center_x, text_center_y]) - np.array([field_center_x, field_center_y]))
		
		if not intersects(coords, text_region) and \
				(field_x_start <= x_end and field_x_end >= x_start and field_center_y >= text_center_y
				 or
				 field_y_start <= y_end and field_y_end >= y_start and field_center_x >= text_center_x):
			image_region = doc_image[y_start-buffer:y_end+buffer, x_start-buffer:x_end+buffer]
			image_region = convert_to_grayscale(image_region)
			image_region = thresholding(image_region)
			
			field_term = pytesseract.image_to_string(image_region, config=ocr_config).replace("\n\x0c", '')
			
			closest_regions.append({"distance": dist, "field_term": field_term, "region": text_region})
			
			if show:
				doc_image = cv2.rectangle(doc_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
			
	closest_regions = sorted(closest_regions, key=lambda x: x["distance"])
	longest_substring = ''
	result = ''
	
	for region in closest_regions:
		for field_name in possible_fields:
			substring = longest_common_substring(field_name.strip().lower(), region["field_term"].strip().lower())
			
			if len(substring) > len(longest_substring):
				longest_substring = substring
				result = field_name
	
	if show:
		cv2.imshow("Closest regions", doc_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
				
	print("Closest field:", result)
	return result
		
	
if __name__ == "__main__":
	doc_image_path = "data/image1.png"
	doc_image = cv2.imread(doc_image_path)
	closest_field_name(doc_image, "Date", [[1390, 578], [1526, 598]], [
		"Tax",
		"Address"
		"Name",
		"Surname",
		"Invoice Date",
		"Due Date",
		"Billing Date",
		"Shipping Date"
	], show=True)
	
	doc_image_path = "data/image2.png"
	doc_image = cv2.imread(doc_image_path)
	closest_field_name(doc_image, "Address", [[440, 635], [705, 750]], [
		"Tax",
		"Name",
		"Surname",
		"Home Address",
		"Business Address",
		"Billing Address",
		"Shipping Address"
	], show=True)
