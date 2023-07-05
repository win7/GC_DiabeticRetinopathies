# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy

import pandas as pd
import shutil
import sys

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image

from tqdm import tqdm
import memory_profiler  # conda install -c anaconda memory_profiler
# cudnn.benchmark = True
# plt.ion()   # interactive mode
import neptune

from tempfile import TemporaryDirectory


def move_split(df, type): # change this function according your dataset
    if not os.path.isdir("dataset"):
        os.makedirs("dataset")

        for item in np.unique(df.iloc[:, -1]):
            os.makedirs("dataset/train/class_{}".format(item))
            os.makedirs("dataset/val/class_{}".format(item))
    
    for index, row in tqdm(df.iterrows()):
        s = "source/train/{}_{}.jpg".format(row["ID"], row["location"])
        t = "dataset/{}/class_{}".format(type, row["level"])
        shutil.copy(s, t)

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

# set initialize weight for custom model
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def classifier_model(num_ftrs, num_classes, init_weight=False):
    # Custom model
    """ torch.nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((2, 2)),
    torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((2, 2)),
    torch.nn.Flatten(), """

    model = torch.nn.Sequential(torch.nn.Linear(num_ftrs, 128),
                                torch.nn.ReLU(),

                                torch.nn.Linear(128, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.5),

                                torch.nn.Linear(64, num_classes))
    if init_weight:
        model.apply(init_weights)
    return model

def initialize_model(model_name, num_classes, feature_extract):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model = None
    input_size = 0

    if model_name == "alexnet":
        model = models.alexnet(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "convnext_large":
        model = models.convnext_large(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 480

    elif model_name == "googlenet":
        model = models.googlenet(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 384

    elif model_name == "inception_v3":
        model = models.inception_v3(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

        """ set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 299 """

    elif model_name == "mnasnet1_3":
        model = models.mnasnet1_3(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "maxvit_t":
        model = models.maxvit_t(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[5].in_features
        model.classifier[5] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "regnet_y_128gf":
        model = models.regnet_y_128gf(weights="IMAGENET1K_SWAG_E2E_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 384

    elif model_name == "resnext101_64x4d":
        model = models.resnext101_64x4d(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        model = models.resnet152(weights="IMAGENET1K_V2")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "shufflenet_v2_x2_0":
        model = models.shufflenet_v2_x2_0(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        input_size = 224

    elif model_name == "swin_v2_b":
        model = models.swin_v2_b(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.head.in_features
        model.head = classifier_model(num_ftrs, num_classes, init_weight=True) # # nn.Linear(num_ftrs, num_classes)
        input_size = 256

    elif model_name == "vgg19":
        model = models.vgg19_bn(weights="IMAGENET1K_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vit_h_14":
        model = models.vit_h_14(weights="IMAGENET1K_SWAG_E2E_V1")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.heads.head.in_features
        model.heads.head = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs,num_classes)
        input_size = 518

    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(weights="IMAGENET1K_V2")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # ---    
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = classifier_model(num_ftrs, num_classes, init_weight=True) # nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 #
        model = models.inception_v3(weights=weights, progress=True)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(run, model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        loop_obj = tqdm(range(num_epochs))
        for epoch in loop_obj:
            # print(f'Epoch {epoch}/{num_epochs - 1}')
            # print('-' * 10)
            loop_obj.set_description(f"Epoch: {epoch + 1}")

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                loop_obj.set_postfix_str(f"{phase}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                run["{}/accuracy".format(phase)].append(epoch_acc)
                run["{}/loss".format(phase)].append(epoch_loss)

                if phase == "train":
                    train_accuracies.append(epoch_acc)
                    train_losses.append(epoch_loss)
                else:
                    val_accuracies.append(epoch_acc)
                    val_losses.append(epoch_loss)

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        run["train/runtime"] = time_elapsed // 60

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, train_accuracies, train_losses, val_accuracies, val_losses

def train_model_v1(run, model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    val_acc_history = []

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loop_obj = tqdm(range(num_epochs))
    for epoch in loop_obj:
        # print(f"Epoch {epoch + 1}/{num_epochs}")
        # print("-" * 10)
        loop_obj.set_description(f"Epoch: {epoch + 1}")

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                train_accuracies.append(epoch_acc)
                train_losses.append(epoch_loss)

                run["train/accuracy"].append(epoch_acc)
                run["train/loss"].append(epoch_loss)
            else:
                val_accuracies.append(epoch_acc)
                val_losses.append(epoch_loss)

                run["val/accuracy"].append(epoch_acc)
                run["val/loss"].append(epoch_loss)

            # print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            loop_obj.set_postfix_str(f"{phase}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    run["train/runtime"] = time_elapsed

    # load best model weights
    model.load_state_dict(best_model_wts)

    # run.stop()
    return model, train_losses, val_losses, train_accuracies, val_accuracies # val_acc_history

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def show_results(train_losses, val_losses, train_accuracies, val_accuracies):
    train_accuracies_ = [h.cpu().numpy() for h in train_accuracies]
    val_accuracies_ = [h.cpu().numpy() for h in val_accuracies]

    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    ax1, ax2 = axes
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses, label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid()
    ax2.plot(train_accuracies_, label="train")
    ax2.plot(val_accuracies_, label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.grid()
    for ax in axes: ax.legend()

def get_predictions(model_ft, dataloaders, data):
    labels = []
    predictions = []
    predictions_proba = []

    with torch.no_grad():
        for batch_x_test, batch_y_test in dataloaders[data]:
            batch_x_test = batch_x_test.to(device).to(torch.float32)
            batch_y_test = batch_y_test.to(device).to(torch.int64)

            batch_test_predictions = model_ft(batch_x_test)
            # batch_test_predictions = model_conv(batch_x_test)

            batch_test_predictions = torch.nn.functional.softmax(batch_test_predictions, dim=-1)
            predictions_proba.append(batch_test_predictions)
            # print(batch_test_predictions)

            batch_test_predictions = batch_test_predictions.max(dim=1).indices
            # print(batch_test_predictions)

            labels.append(batch_y_test)
            predictions.append(batch_test_predictions)

    labels = torch.cat(labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    predictions_proba = torch.cat(predictions_proba, dim=0)

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    predictions_proba = predictions_proba.cpu().numpy()
    return labels, predictions, predictions_proba