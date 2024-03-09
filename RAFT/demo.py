import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'#'cpu'#

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # print("shape of their array",img.dtype)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    print(type(flo))
    image_path = f"./outputs/{i:03d}.png"
    # cv2.imwrite(image_path,img_flo[:, :, [2,1,0]]/255.0)
    # final_flo = flo / 255.0
    cv2.imwrite(image_path,flo)

    # plt.imsave(final_flo,image_path) 
    # import matplotlib.pyplot as plt
    # plt.imshow(flo / 255.0)
    # plt.show()
    # plt.imshow(img_flo[:, :, [2,1,0]]/255.0)
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    # print("print here",args.model)
    # model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        i = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            ourframe = cv2.imread(imfile1)
            ourframe_second = cv2.imread(imfile2)
            # print("shape of our cv2 frame",ourframe.dtype)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            #--------------------------------------
            img = torch.from_numpy(ourframe).permute(2, 0, 1).float()
            ourframe = img[None].to(DEVICE)
            img = torch.from_numpy(ourframe_second).permute(2, 0, 1).float()
            ourframe_second = img[None].to(DEVICE)
            print("shape of our cv2 frame",ourframe.shape)
            print("Their new image",image1.shape)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            padderours = InputPadder(ourframe.shape)
            ourframe, ourframe_second = padder.pad(ourframe, ourframe_second)
            print("our padding",ourframe.shape)
            print("padded their",image1.shape)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up,i)
            i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
