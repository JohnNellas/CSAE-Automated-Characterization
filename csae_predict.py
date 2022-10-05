import torch
import torchvision
import argparse
import csae_module as cm
import os
import json
import numpy as np
import pandas as pd


# =========================================
# Parse Arguments
# =========================================

parser = argparse.ArgumentParser(
    description="Automated Skin Lesion Characterization and COVID-19 Diagnosis from CT Scans using Convolutional Supervised Autoencoders.")

parser.add_argument("-p", "--path", required=True, metavar="Path", type=str,
                    help="Path to the directory containing the dataset.")
parser.add_argument("-d", "--dataset", required=True, metavar="Dataset", type=str,
                    choices=["covid", "skin"], help=f"Type of data contained in the dataset. Available choices: {', '.join(['covid', 'skin'])} (covid: COVID-19 CT Scans, skin: Skin Lesions).")
parser.add_argument("-l", "--latent-space-dimensions", required=True, metavar="Latent Space Dimensions", type=int,
                    choices=[2, 10], help=f"The number of dimensions in the latent space. Available choices: {', '.join(['2', '10'])}.")
parser.add_argument("-b", "--batch-size", required=False, metavar="Batch Size", type=int,
                    default=64, help="The Batch Size for data loading (Not Required Argument, default value is 64).")
args = parser.parse_args()

# use the appropriate device (cpu or gpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")

batch_size = args.batch_size
latent_space_dimensions = args.latent_space_dimensions
num_conv_blocks = 1
code_word = "best_model_linear_1cb_relu"
activation_encoder, activation_decoder, activation_classifier = "relu", "relu", "relu"

# =========================================
# Dataset Loading and preprocessing
# =========================================

if args.dataset == "covid":
    preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),  # convert to grayscale
        # resize image to a smaller size
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])
else:
    preprocessing = torchvision.transforms.Compose([
        # resize image to a smaller size
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])

entire_dataset = cm.load_dataset_mod.LoadDataset(
    root_dir=args.path, transform=preprocessing)

image_shape = list(entire_dataset[0][0].shape)
image_shape = image_shape[1:] + [image_shape[0]]

# Let's wrap a dataloader around the dataset
testing_dataloader = torch.utils.data.DataLoader(entire_dataset,
                                                 batch_size=batch_size)


checkpoint_parent_path = os.path.join(".", f"checkpoints_csae_{code_word}")
checkpoint_path = os.path.join(
    checkpoint_parent_path, f"checkpoints_ae_classification_{args.dataset}_ls_{latent_space_dimensions}_dimensions")

with open(os.path.join(checkpoint_path, f"class_info.json"), "r") as read_file:
    class_info = json.load(read_file)

class_info = {k: v for v, k in class_info.items()}
num_classes = len(class_info.keys())

# =============================================================================
# Creating and Loading the best model for the dataset
# =============================================================================

# set up the models
encoder = cm.Encoder(image_shape,
                     latent_space_dims=latent_space_dimensions,
                     activation_name=activation_encoder,
                     num_conv_blocks=num_conv_blocks).to(device)

classifier = cm.Classifier(latent_space_dims=latent_space_dimensions,
                           number_of_classes=num_classes,
                           activation_name=activation_classifier).to(device)

decoder = cm.Decoder(image_shape,
                     enc_out=latent_space_dimensions,
                     activation_name=activation_decoder,
                     num_conv_blocks=num_conv_blocks).to(device)

csae = cm.CSAE(encoder, decoder, classifier).to(device)

csae.load_state_dict(torch.load(os.path.join(
    checkpoint_path, f"best_weights_{args.dataset}.pth")))

print("Model Created and Weight are Loaded...")
# =============================================================================
# Inference
# =============================================================================
print("Inferencing...")
total_logits = list()
filenames = list()
with torch.no_grad():
    for batch, (X, bfilenames) in enumerate(testing_dataloader):

        # send the batch to the gpu
        X = X.to(device)

        filenames.extend(bfilenames)

        # calculate the network outputs
        _, logits, _ = csae(X)

        total_logits.append(logits.to("cpu").detach().numpy())

total_logits = np.concatenate(total_logits, axis=0)
predictions = np.argmax(total_logits, axis=1).astype("int")

# =============================================================================
# Directory Tree Construction for the Results
# =============================================================================
classification_results_parent_path = os.path.join(
    ".", f"classification_results_csae")
if not os.path.isdir(classification_results_parent_path):
    os.mkdir(classification_results_parent_path)

classification_results_middle_path = os.path.join(
    classification_results_parent_path, f"classification_results_{args.dataset}")
if not os.path.isdir(classification_results_middle_path):
    os.mkdir(classification_results_middle_path)

classification_results_path = os.path.join(
    classification_results_middle_path, f"classification_results_ls_{latent_space_dimensions}_dimensions")
if not os.path.isdir(classification_results_path):
    os.mkdir(classification_results_path)
    

# =============================================================================
# Save the results to the destination folder
# =============================================================================
df = pd.DataFrame({"Filename": filenames, "Prediction": predictions})
df["Prediction"] = df["Prediction"].map(class_info)
df.to_excel(os.path.join(classification_results_path, "results.xlsx"), index=False)