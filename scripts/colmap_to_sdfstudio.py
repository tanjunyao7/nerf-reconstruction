import argparse
import sys
import struct

import numpy as np
import glob
import os
import cv2
import shutil
from joblib import Parallel, delayed
import subprocess
from read_write_model import *
import json


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def main(model_dir,image_dir):
    cameras, images, points3D = read_model(model_dir)
    cov_images = {}
    for p in points3D.values():
        image_ids = p.image_ids.tolist()
        for i in image_ids:
            assert isinstance(i,int)
            for j in image_ids:
                assert isinstance(j,int)
                if i==j:continue

                if i not in cov_images:
                    cov_images[i]=[]
                cov_images[i].append(j)

                if j not in cov_images:
                    cov_images[j]=[]
                cov_images[j].append([i])

    with open(os.path.join(image_dir,'pairs.txt'),'w') as file:
        for k,v in cov_images.items():
            file.write(images[k].name)
            for i in v:
                if isinstance(i,list):
                    i = i[0]
                file.write(' '+images[i].name)
            file.write('\n')

    for k,v in cameras.items():
        print(v)
        assert v.model == 'PINHOLE'
        fx,fy,cx,cy =v.params[:4]
        width = v.width
        height = v.height

    K = np.array([[fx,0,cx,0],
                  [0,fy,cy,0],
                  [0,0,1,0],
                  [0,0,0,1]]).tolist()

    metadata = {
          'camera_model': 'OPENCV', # camera model (currently only OpenCV is supported)
          'height': height, # height of the images
          'width': width, # width of the images
          'has_mono_prior': False, # use monocular cues or not
            "has_foreground_mask": False,
          'pairs': 'pairs.txt', # pairs file used for multi-view photometric consistency loss
          'worldtogt': [
              [1, 0, 0, 0], # world to gt transformation (useful for evauation)
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
            ],
          'scene_box': {
              'aabb': [
                  [-1, -1, -1], # aabb for the bbox
                  [1, 1, 1],
                ],
              'near': 0.5, # near plane for each image
              'far': 4.5, # far plane for each image
              'radius': 1.0, # radius of ROI region in scene
              'collider_type': 'near_far'
            },
        'frames':[]
        }


    for k,v in images.items():
        filename = v.name


        rotation = v.qvec2rotmat()
        translation = v.tvec.reshape(3, 1)*0.1
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        # c2w[0:3, 1:3] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1

        frame = {'rgb_path':filename,
                 'intrinsics':K,
                'camtoworld':c2w.tolist()
                 }
        metadata['frames'].append(frame)

    with open(os.path.join(image_dir,'meta_data.json'),'w') as f:
        json.dump(metadata,f)


if __name__ == '__main__':
    model_dir = sys.argv[1]
    image_dir = sys.argv[2]
    main(model_dir,image_dir)
