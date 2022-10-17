from typing import List

import cv2
import pytesseract


def locate_text_regions(image_path: str, show=False) -> List[List[List[float]]]:
	"""
	Find all text regions in the image.
	:param image_path: Path to the document image.
	:param show: Parameter indicating whether to show the detection results or not.
	:return: A list of all detected text regions represented as [[x_start, y_start], [x_end, y_end]].
	"""
	image = cv2.imread(image_path)
	boxes = pytesseract.image_to_data(image)
	boxes = boxes.split('\n')[1:]
	regions = []
	
	for row in boxes:
		row = row.split('\t')
		
		if len(row[-1].strip()) == 0:
			continue
		
		x, y, w, h = list(map(int, row[6:10]))
		regions.append([[x, y], [x+w, y+h]])
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		
	if show:
		cv2.imshow("Text regions", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	return regions
