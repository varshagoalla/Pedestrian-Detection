import cv2, sys, os, argparse, json, numpy as np
from itertools import groupby
from operator import itemgetter

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root')
	parser.add_argument('--test')
	parser.add_argument('--out')
	args = parser.parse_args()
	return args


def nms(boxes, overlapThresh):
	if len(boxes) == 0:
		return []
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2] + x1
	y2 = boxes[:,3] + y1
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
	return boxes[pick]

def main():
	args = parse_args()
	d = []
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	f = open(args.test)
	data = json.load(f)
	
		
	for item in data['images']:
		image_path = args.root + "/" + item['file_name']
		image = cv2.imread(image_path)
		#img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects, weights = hog.detectMultiScale(image, winStride=(1,1), padding=(8, 8), scale=1.05)
		rects = np.array([np.append(rects[i],weights[i]) for i in range(len(rects))])
		for [x,y,w,h,weight] in nms(rects,0.5):
			cv2.rectangle(image, (int(x),int(y)), (int(x+w),int(y+h)),(0, 255, 0), 2)
			d.append({'image_id': int(item['id']),'category_id': 1,'bbox': [int(x),int(y),int(w),int(h)],'score': float(weight)})
		cv2.imshow('HOG detection', image)
		cv2.waitKey(30)
	with open(args.out, "w") as outfile:
		json.dump(d,outfile)
main()


#https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/?_ga=2.77179990.216516709.1636073201-2015776852.1632591078
