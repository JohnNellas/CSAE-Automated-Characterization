# Automated Characterization using Convolutional Supervised Autoencoders
Automated Skin Lesion Characterization and COVID-19 Diagnosis from CT Scans using [Convolutional Supervised Autoencoders](https://arxiv.org/abs/2208.12152).

## Usage

1. Clone the repository to local machine.
```
git clone https://github.com/JohnNellas/CSAE-Automated-Characterization
```
2. Create the environment from the ```environment.yml``` file.
```
conda env create -f environment.yml
```
3. Activate the newly created environment.
```
conda activate csae_env
```
4. See the script documentation.
```
python3 csae_predict.py -h

usage: csae_predict.py [-h] -p Path -d Dataset -l Latent Space Dimensions [-b Batch Size]

Automated Skin Lesion Characterization and COVID-19 Diagnosis from CT Scans using Convolutional Supervised Autoencoders.

optional arguments:
  -h, --help            show this help message and exit
  -p Path, --path Path  Path to the directory containing the dataset.
  -d Dataset, --dataset Dataset
                        Type of data contained in the dataset. Available choices: covid, skin (covid: COVID-19 CT Scans, skin: Skin Lesions).
  -l Latent Space Dimensions, --latent-space-dimensions Latent Space Dimensions
                        The number of dimensions in the latent space. Available choices: 2, 10.
  -b Batch Size, --batch-size Batch Size
                        The Batch Size for data loading (Not Required Argument, default value is 64).
```

## Example
Utilize a pretrained CSAE with a 2 dimensional latent space to perform Automated Skin Lesion Characterization (```skin``` option) for the images contained in the directory ```/path/to/images```.

```
python3 csae_predict -p /path/to/images -d skin -l 2
```

Employ a pretrained CSAE with a 2 dimensional latent space to perform Automated COVID-19 diagnosis from CT-Scans (```covid``` option) for the images contained in the directory ```/path/to/images```.

```
python3 csae_predict -p /path/to/images -d covid -l 2
```

## Execution Details

The script execution predicts the label of each image contained in the specified as argument directory, and creates an excel file that contain the results as pairs of filenames and predicted labels. This excel file is stored under the ```classification_results_csae/classification_results_{datasetName}/classification_results_ls_{latentSpaceDimensions}_dimensions```, with the filename ```results.xlsx```, where the required directories are automatically created during the script execution.

## Acknowledgements
This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI), under grant agreement No 1901.


