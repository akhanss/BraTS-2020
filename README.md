## A complete pipeline for BraTS 2020: [Multimodal Brain Tumor Segmentation Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/) based on 3D U-net

The github repo lets you train a 3D U-net model using [BraTS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html) (perhaps it can be used for previous BraTS dataset). While this repo is a ready-to-use pipeline for segmentation task, one may extend this repo for other tasks such as survival task and Uncertainty task. Even the repo may be used for other 3D dataset/task.

If you face any problem, please feel free to open an issue.

## Directory structure

```
.
├─ data
  ├─ brats20	# Data provided by the BraTS 2020 competition host
    ├─ TrainingData
	├─ BraTS20_Training_001
	    ├─ BraTS20_Training_001_flair.nii.gz
	    ├─ BraTS20_Training_001_seg.nii.gz
	    ├─ BraTS20_Training_001_t1.nii.gz
	    ├─ BraTS20_Training_001_t1ce.nii.gz
	    ├─ BraTS20_Training_001_t2.nii.gz
	├─ BraTS20_Training_002
	├─ ...
    ├─ ValidationData
	├─ BraTS20_Validation_001
	    ├─ BraTS20_Validation_001_flair.nii.gz
	    ├─ BraTS20_Validation_001_t1.nii.gz
	    ├─ BraTS20_Validation_001_t1ce.nii.gz
	    ├─ BraTS20_Validation_001_t2.nii.gz
	    ├─ ...
├─ model	# Generated training and validation split (training_ids.pkl, validation_ids_pkl, test_ids.pkl), processed data file (brats20_data.h5, brats20_data_test.h5), and save best training model (isensee_2017_model.h5)
├─ output	# Generated prediction file
├─ src		# Souce code
    ├─ unet3d
    ├─ config.py
    ├─ inference.py
    ├─ train.py
├─ inference.sh
├─ train.sh
```

## Packages
- Python >= 3.5 (my current version is 3.7.7)
- tensorflowgpu==1.15 (other 1.x version should work)
- Other packages: pytables, SimpleITK, nilearn, nibabel
- Optional package: nipype (For n4itk bias correction preprocessing only. However, I didn't achieve that much performance gain using this technique!)

## How to run
If you prepare directory structure properly, you are done!

### Train:
~~~
$ ./train.sh
~~~
### Validation/Test:
~~~
$ ./inference.sh
~~~

## Results on validation set
Label		Dice_ET	Dice_WT	Dice_TC	Sensitivity_ET	Sensitivity_WT	Sensitivity_TC	Specificity_ET	Specificity_WT	Specificity_TC	Hausdorff95_ET	Hausdorff95_WT	Hausdorff95_TC

Mean		0.60954	0.82701	0.74769	0.66736	0.86277	0.75625	0.99939	0.99834	0.99931	46.96773	10.70035	11.63979

StdDev		0.31934	0.15026	0.21439	0.33152	0.14198	0.22683	0.00098	0.00127	0.00108	112.51574	17.23635	18.01854

Median		0.7783	0.87372	0.84036	0.81398	0.90829	0.85127	0.99967	0.99875	0.99967	3.60555		5.47723		5.74456

25quantile	0.44618	0.814	0.63882	0.5355	0.83669	0.64064	0.99921	0.998	0.99928	2		3.74166		3

75quantile	0.84443	0.90629	0.89807	0.89286	0.94265	0.91698	0.99987	0.99916	0.99987	15.67788	10.44031	12.36932


## Acknowledgment
Significant code has been borrowed from [ellisdg's repository](https://github.com/ellisdg/3DUnetCNN) which is based on [Isensee et al.'s paper](https://doi.org/10.1007/978-3-030-11726-9_21). I really appreciate David G Ellis's contributions to the community.
