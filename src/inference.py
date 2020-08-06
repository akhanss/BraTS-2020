import numpy as np

import os
import glob

from unet3d.data_test import write_data_to_file, open_data_file
from unet3d.prediction import run_validation_cases
from unet3d.utils.utils import pickle_dump, pickle_load

from config_test import config
    
def fetch_testing_data_files(return_subject_ids=False):
    testing_data_files = list()
    subject_ids = list()
    # processed_dir = config["preprocessed_test"]
    processed_dir = config["test_dir"]
    for idx, subject_dir in enumerate(glob.glob(os.path.join(processed_dir, "*"))):
        #if idx == 2: break
        subject_ids.append(os.path.basename(subject_dir))
        # print(subject_dir, subject_ids)
        subject_files = list()
        for modality in config["training_modalities"]:
            subject_files.append(os.path.join(subject_dir, os.path.basename(subject_dir) + '_' + modality + ".nii.gz"))
        testing_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return testing_data_files, subject_ids
    else:
        return testing_data_files
    

def main(overwrite=False):
    
    # convert test images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file_test"]):
        testing_files, subject_ids = fetch_testing_data_files(return_subject_ids=True)

        #write_data_to_file(testing_files, config["data_file_test"], image_shape=config["image_shape"],
                           #subject_ids=subject_ids)
        write_data_to_file(testing_files, config["data_file_test"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)
    
    test_keys_file=config["test_file"]
    num_test_files = config["num_test_files"] # Change this accordingly on config.py
    pickle_dump(list(np.arange(num_test_files)), test_keys_file)

    data_file_opened = open_data_file(config["data_file_test"])
    
    prediction_dir = config["output_dir"]
    run_validation_cases(test_keys_file,
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file_test"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
