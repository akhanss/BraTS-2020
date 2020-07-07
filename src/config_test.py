import os

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "t2", "flair"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 5  # cutoff the training after this many epochs, default 500
config["patience"] = 3  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 2  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-4
config["learning_rate_drop"] = 0.1  # factor by which the learning rate will be reduced
config["validation_split"] = 0.9  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = 0.25  # switch to None if you want no distortion
config["augment"] = True # config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

# To be changed onwards

'''
config["img_dir"] = "/data/train/image"
config["label_dir"] = "/data/train/label"
config["test_dir"] = "/data/test"
config["output_dir"] = "/data/output"
config["preprocessed"] = "/data/model/preprocessed"
'''
config["img_dir"] = "/home/nas1_userC/a_khanss/dataset/MICCAI_BraTS2020/TrainingData"
# config["label_dir"] = "/home/nas1_userC/a_khanss/dataset/MICCAI_BraTS2020/TrainingData"
config["test_dir"] = "/home/nas1_userC/a_khanss/dataset/MICCAI_BraTS2020/ValidationData"
config["output_dir"] = "./data/output"
# config["preprocessed"] = "/home/dataset/med-img/brats/2018/preprocessed"
# '''
# config["preprocessed_test"] = "/home/dataset/med-img/brats/2018/preprocessed_test"
config["test_file"] = "./data/model/test_ids.pkl"
config["num_test_files"] = 125 # Change it to number of test files

config["data_file"] = "./data/model/brats2018_data.h5"
config["data_file_test"] = "./data/model/brats2018_data_test.h5"
config["model_file"] = "./data/model/isensee_2017_model.h5"
config["training_file"] = "./data/model/isensee_training_ids.pkl"
config["validation_file"] = "./data/model/isensee_validation_ids.pkl"
config["test_file"] = "./data/model/isensee_test_ids.pkl"
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
