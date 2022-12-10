from typing import List

import numpy as np
import pytesseract


def locate_text_regions(image: np.array) -> List[List[List[float]]]:
	"""
	Find all text regions in the image.
	:param image: The document image represented as numpy array.
	:return: A list of all detected text regions represented as [[x_start, y_start], [x_end, y_end]].
	"""
	boxes = pytesseract.image_to_data(image)
	boxes = boxes.split('\n')[1:]
	regions = []
	
	for row in boxes:
		row = row.split('\t')
		if len(row[-1].strip()) == 0:
			continue
		x, y, w, h = list(map(int, row[6:10]))
		regions.append([[x, y], [x+w, y+h]])
		
	return regions
