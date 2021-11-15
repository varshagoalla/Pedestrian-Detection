import cv2, sys, os, argparse, json, numpy as np, torchvision, torch
from itertools import groupby
from operator import itemgetter
from PIL import Image
import torchvision.transforms as T
from torch import utils

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--test')
    parser.add_argument('--out')
    args = parser.parse_args()
    return args

def to_list(v):
    l = []
    for item in v:
        l.append(item['bbox'])
    return l

class PennFudanDataset(object):

    def __init__(self, root, json_file):
        self.root = root
        self.data = json.load(open(json_file))
        self.imgs = list(sorted(self.data['images'], key=lambda d: d['id']))

    def __getitem__(self, idx):
        
        img_path = self.root + "/" + self.imgs[idx]["file_name"]
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        image_id = torch.tensor(idx)
        return img, image_id


    def __len__(self):
        return len(self.imgs)

def get_prediction(img):
    pred = model([img])
    pred_class = ['person' if i==1 else 'non-person' for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[i[0], i[1],i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_boxes = [pred_boxes[i]  for i in range(len(pred_class)) if pred_class[i]=='person']
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_score = [pred_score[i]  for i in range(len(pred_class)) if pred_class[i]=='person']
    pred_boxes = [pred_boxes[i]+[pred_score[i]] for i in range(len(pred_boxes))]
    return np.array(pred_boxes)

def main():
    args = parse_args()
    d = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PennFudanDataset(args.root,args.test)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for (img, img_id) in data_loader:
        img = img[0]
        boxes = get_prediction(img)
        image = np.asarray(T.ToPILImage()(img))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        boxes = nms(boxes,0.6)
        for i in range(len(boxes)): 
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            score = float(box[4])
            cv2.rectangle(image, (x, y), (x2, y2),(0, 255, 0), 2)
            d.append({'image_id': int(img_id),'category_id': 1,'bbox': [x,y,x2-x,y2-y],'score': score})
        cv2.imshow('RCNN detection', image)
        cv2.waitKey(1000)
        
    with open(args.out, "w") as outfile:
        json.dump(d,outfile)


main()
