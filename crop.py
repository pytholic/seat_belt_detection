import cv2
import os, glob

path = './dummy_data/original'
out_path = './dummy_data/cropped/'

for idx, img in enumerate(glob.glob(path + '/*.jpg')):
	img = cv2.imread(img)

	# images are 1080*1920
	size_dict ={
		'less':  img[200:900, 200:1700],
		'med': img[350:750, 400:1400],
		'extreme': img[400:700, 500:1300]
	}

	crop = size_dict['extreme'] 

	# # cv2.imshow('original', img)
	# # cv2.imshow('cropped', crop)
	# # cv2.waitKey(0)
	# # cv2.destroyAllWindows()

	cv2.imwrite(os.path.join(out_path, f'{idx}.jpg'), crop)
