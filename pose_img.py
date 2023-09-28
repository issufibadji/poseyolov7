import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from time import time
import os
import argparse

MODEL_PATH = 'yolov7-w6-pose.pt'

#Entrada de modelo e cpu
def main_poses(image_input:str, size_image:int, path_result:str):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(MODEL_PATH, map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    #Leitura da imagem  
    image = cv2.imread(image_input)
    image = letterbox(image, size_image, stride=64, auto=True)[0]
    #image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    
    #Inferencia de modelo
    t_model = time()
    if torch.cuda.is_available():
        image = image.half().to(device)   
    
    #estimativa de pontos-chaves
    t_keypoints = time()
    with torch.no_grad():
        output, _ = model(image)
        print("Time model: {:.3f}".format(time()-t_model))
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
    print("Time keypoints:{:.3f}".format(time()-t_keypoints))

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
   
    #cv2.imshow('Saida-Pontos-chave-Corpo', nimg)
    #cv2.imwrite("Saida-Pontos-chave-Corpo.jpg", nimg)
    # Skeleton plot
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    print("Total time:{:.3f}".format(time()-t_model))
    
    # plt.figure(figsize=(8,8))
    # plt.axis('off')
    # plt.imshow(nimg)
    # plt.savefig('resul.png')
    #cv2.imshow('Saida-esqueleto', nimg)
    img_output_path = os.path.join(path_result, os.path.basename(image_input))
    cv2.imwrite(img_output_path, cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Input parameters')
    group.add_argument('--img_input', type=str, required=False, default=os.path.join("src", "images", "img1.png"), help=f'Image Path.')
    group.add_argument('--img_size', type=int, required=False, default=640, help=f'Image Path.')
    group.add_argument('--path_result', type=str, required=False, default=os.path.join("src", "result"), help=f'Image Path.')

    args = parser.parse_args()
    main_poses(args.img_input, args.img_size, args.path_result)
    print("finish")
    # EXAMPLE: python pose_img.py --img_input src/images/img10.jpg