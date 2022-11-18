from typing import List

import cv2
import numpy as np
import pytesseract


def locate_text_regions(image: np.array, show=False) -> List[List[List[float]]]:
	"""
	Find all text regions in the image.
	:param image: The document image represented as numpy array.
	:param show: Parameter indicating whether to show the detection results or not.
	:return: A list of all detected text regions represented as [[x_start, y_start], [x_end, y_end]].
	"""
	show_image = image.copy()
	boxes = pytesseract.image_to_data(image)
	boxes = boxes.split('\n')[1:]
	regions = []
	
	for row in boxes:
		row = row.split('\t')
		
		if len(row[-1].strip()) == 0:
			continue
		
		x, y, w, h = list(map(int, row[6:10]))
		regions.append([[x, y], [x+w, y+h]])
		cv2.rectangle(show_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		
	if show:
		cv2.imshow("Text regions", show_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	return regions
