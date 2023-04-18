import argparse
import glob
import os
import shutil
import subprocess

import pycolmap

import colmap_to_sdfstudio


def sparse_reconstruction(image_dir, output_dir):
    sparse_model_dir = os.path.join(output_dir, 'sparse')

    if os.path.exists(sparse_model_dir):
        shutil.rmtree(sparse_model_dir)
    os.makedirs(sparse_model_dir)

    database = os.path.join(sparse_model_dir, 'db.db')

    pycolmap.extract_features(database, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model='OPENCV')

    pycolmap.match_exhaustive(database, sift_options={'num_threads': 16})

    pycolmap.incremental_mapping(
        database, image_dir, sparse_model_dir, options={'num_threads': 16})

    undist_dir = os.path.join(output_dir, 'undist')
    pycolmap.undistort_images(undist_dir, os.path.join(sparse_model_dir, '0'), image_dir)

    sparse_model_dir = os.path.join(undist_dir, 'sparse')
    undist_image_dir = os.path.join(undist_dir, 'images')

    return sparse_model_dir, undist_image_dir


def dense_reconstruction(image_dir, colmap_model_dir, output_dir):
    colmap_to_sdfstudio.main(colmap_model_dir, image_dir)

    nerf_output_dir = os.path.join(output_dir, 'sdfstudio')
    subp = subprocess.Popen(['ns-train', 'nerfacto',
                             '--output-dir', nerf_output_dir,
                             '--pipeline.model.predict-normals', 'True',
                             '--trainer.max-num-iterations', '10000',
                             '--viewer.quit-on-train-completion', 'True',
                             '--viewer.skip-openrelay', 'True',
                             'sdfstudio-data',
                             '--data', image_dir])
    subp.wait()

    config_file = glob.glob(nerf_output_dir + '/**/config.yml', recursive=True)[0]

    model_output_dir = os.path.join(output_dir, 'dense')
    subp = subprocess.Popen(['ns-export', 'pointcloud',
                             '--load-config', config_file,
                             '--output-dir', model_output_dir,
                             '--num-points', '10000000',
                             '--remove-outliers', 'True',
                             '--use-bounding-box', 'True',
                             '--estimate-normals', 'False',
                             '--bounding-box-min', '-1', '-1', '-1',
                             '--bounding-box-max', '1', '1', '1',
                             '--std-ratio', '1'])
    subp.wait()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder",
                        help="folder of images", type=str, required=True)
    parser.add_argument("-o", "--output_folder",
                        help="dense reconstruction output folder", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_folder
    output_dir = args.output_folder

    sparse_model_dir, undist_image_dir = sparse_reconstruction(image_dir, os.path.join(output_dir, 'sparse'))
    dense_reconstruction(undist_image_dir, sparse_model_dir, os.path.join(output_dir))
