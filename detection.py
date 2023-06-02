# import dataset
from datasets import load_dataset
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.patches as patches
from tqdm.auto import tqdm
from PIL import Image
import os

# data = load_dataset(
#     "jamescalam/image-text-demo",
#     split="train",
#     revision="180fdae"
# )

# transform the image into tensor
transform = transforms.ToTensor()
input_image = Image.open('IMG_4547.jpg')

img = transform(input_image)


# define processor and model
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)





colors = ['#FAFF00', '#8CF1FF']

def get_patches(img, patch_size=256):
    # add extra dimension for later calculations
    img_patches = img.data.unfold(0,3,3)
    # break the image into patches (in height dimension)
    img_patches = img_patches.unfold(1, patch_size, patch_size)
    # break the image into patches (in width dimension)
    img_patches = img_patches.unfold(2, patch_size, patch_size)
    return img_patches


def get_scores(img_patches, prompt, window=6, stride=1):
    patch = 256
    # initialize scores and runs arrays
    scores = torch.zeros(img_patches.shape[1], img_patches.shape[2])
    runs = torch.ones(img_patches.shape[1], img_patches.shape[2])

    # iterate through patches
    for Y in range(0, img_patches.shape[1]-window+1, stride):
        for X in range(0, img_patches.shape[2]-window+1, stride):
            # initialize array to store big patches
            big_patch = torch.zeros(patch*window, patch*window, 3)
            # get a single big patch
            patch_batch = img_patches[0, Y:Y+window, X:X+window]
            # iteratively build all big patches
            for y in range(window):
                for x in range(window):
                    big_patch[y*patch:(y+1)*patch, x*patch:(x+1)*patch, :] = patch_batch[y, x].permute(1, 2, 0)
            inputs = processor(
                images=big_patch, # image trasmitted to the model
                return_tensors="pt", # return pytorch tensor
                text=prompt, # text trasmitted to the model
                padding=True
            ).to(device) # move to device if possible

            score = model(**inputs).logits_per_image.item()
            # sum up similarity scores
            scores[Y:Y+window, X:X+window] += score
            # calculate the number of runs 
            runs[Y:Y+window, X:X+window] += 1
    # calculate average scores
    scores /= runs
    # clip scores
    for _ in range(3):
        scores = np.clip(scores-scores.mean(), 0, np.inf)
    # normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def get_box(scores, patch_size=256, threshold=0.5):
    detection = scores > threshold
    # find box corners
    y_min, y_max = np.nonzero(detection)[:,0].min().item(), np.nonzero(detection)[:,0].max().item()+1
    x_min, x_max = np.nonzero(detection)[:,1].min().item(), np.nonzero(detection)[:,1].max().item()+1
    # convert from patch co-ords to pixel co-ords
    y_min *= patch_size
    y_max *= patch_size
    x_min *= patch_size
    x_max *= patch_size
    # calculate box height and width
    height = y_max - y_min
    width = x_max - x_min
    return x_min, y_min, width, height

def detect(prompts, img, patch_size=256, window=6, stride=1, threshold=0.5):
    # build image patches for detection
    img_patches = get_patches(img, patch_size)
    # print(img_patches.shape)
    X = img_patches.shape[1]
    Y = img_patches.shape[2]

    # convert image to format for displaying with matplotlib
    image = np.moveaxis(img.data.numpy(), 0, -1)
    # initialize plot to display image + bounding boxes
    fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))
    ax.imshow(image)
    # process image through object detection steps
    for i, prompt in enumerate(tqdm(prompts)):
        scores = get_scores(img_patches, prompt, window, stride)
        x, y, width, height = get_box(scores, patch_size, threshold)
        # create the bounding box
        rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor=colors[i], facecolor='none')
        # add the patch to the Axes
        ax.add_patch(rect)
    plt.show()





detect(["Give me the blue box"], img, window=4, stride=1)