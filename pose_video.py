import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import tqdm 
from time import time
import os
import argparse
from typing import Tuple
MODEL_PATH = 'yolov7-w6-pose.pt'

# TODO: Remove -- used only to test
def cap_videoCam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def put_text(img:np.ndarray, text:str, org: Tuple[int,int] = (50, 50) ) -> np.ndarray:
    return cv2.putText(
        img = img,
        text = text,
        org = org,
        fontFace = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 1.0,
        color = (125, 246, 55),
        thickness = 3
    )


def main_poses_video(video_input:str, path_result:str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(MODEL_PATH, map_location=device)
    model = weigths['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    cap = cv2.VideoCapture(video_input) #para video prontas
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    w, h = None, None
    all_result = []
    # FPS do video_input
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get total frames numbers of video_input
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress-bar 
    progress_bar = tqdm.tqdm(range(max_frames), " Process frames..")
    while True:
        init_time = time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        image = frame #cv2.imread('./images/gr2.jpg')
        image = letterbox(image, 640, stride=64, auto=True)[0]
        #image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(device)   
        
        with torch.no_grad():
            output, _ = model(image)
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # Plot skeleton-keypoints
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        # save frame with skeleton-keypoints in a list (all_result) to save in a video after
        nimg = put_text(cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR), f"Time: {(time()-init_time):.4f} sec")
        all_result.append(nimg)
        if w is None: # get width an heigth from frame to use in video generator
            w, h = nimg.shape[1], nimg.shape[0] 
        progress_bar.update(1)
        # When everything done, release the capture
    cap.release()

    # Save video from list images in all_result
    video_input_path = os.path.join(path_result, os.path.basename(video_input))
    out = cv2.VideoWriter(video_input_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in all_result:
        out.write(f)
    out.release()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Input parameters')
    group.add_argument('--video_input', type=str, required=False, default=os.path.join("src", "video", "vid1.mp4"), help=f'Video Path.')
    group.add_argument('--path_result', type=str, required=False, default=os.path.join("src", "result"), help=f'Result Path.')

    args = parser.parse_args()
    main_poses_video(args.video_input, args.path_result)
    print("finish")