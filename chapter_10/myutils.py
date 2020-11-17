import os
import torch
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm_notebook


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_vids(path_to_jpgs):
    list_of_categories = os.listdir(path_to_jpgs)
    ids = []
    labels = []
    for category in list_of_categories:
        path_to_category = os.path.join(path_to_jpgs, category)
        list_of_sub_categories = os.listdir(path_to_category)
        path_to_sub_categories = [os.path.join(path_to_category, los) 
                                    for los in list_of_sub_categories]
        ids.extend(path_to_sub_categories)
        labels.extend([category] * len(list_of_sub_categories))
    return ids, labels, list_of_categories


def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    x = to_pil_image(x)
    return x


def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    validation_data_loader = ["validation_data_loader"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path_to_weights = params["path_to_weights"]

    loss_history = {
            "train": [],
            "val": [],
    }

    metric_history = {
            "train": [],
            "val": [],
    }

    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print("Epoch {}/{}, current lr = {}".format(epoch, num_epochs -1, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func,
                                              train_dl,
                                              sanity_check,
                                              opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()

        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func,
                                              validation_data_loader,
                                              sanity_check,
                                              opt)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path_to_weights)
            print("Copied best model weights!")

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_weights)

        print("Train loss: %.6f, Dev loss: %.6f, Accuracy: %.2f" \
                %(train_loss, val_loss, 100 * val_metric))
        print("-" * 10)

    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history


# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def metrics_batch(output, target):
    prediction = output.argmax(dim=1, keepdim=True)
    corrects = prediction.eq(target.view_as(prediction)).sum().item()


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    #len_data = len(dataset_dl.dataset)
    len_data = len(dataset_dl)

    for xb, yb in tqdm_notebook(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss = running_loss + loss_b

        if metric_b is not None:
            running_metric = running_metric + metric_b
        if sanity_check is True:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)

    return loss, metric


def plot_loss(loss_history, metric_history):
    num_epochs = len(loss_history["train"])

    plt.title("Train-val Loss")
    plt.plot(range(1, num_epochs+1), loss_history["train"], label="train")
    plt.plot(range(1, num_epochs+1), loss_history["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()


from torch import nn

class Resnet18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnet18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dropout_rate = params_model["dropout_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]

        base_model = models.resnet18(pretrained=pretrained)
        num_features = base_model.fc.in_features
        base_model.fc = Identity()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.base_model((x[:, ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))

        for ii in range(1, ts):
            y = self.base_model((x[:, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


from torchvision import models
from torch import nn

def get_model(num_classes, model_type="rnn"):
    if model_type == "rnn":
        params_model = {
                "num_classes": num_classes,
                "dropout_rate": 0.1,
                "pretrained": True,
                "rnn_num_layers": 1,
                "rnn_hidden_size": 100,}
        model = Resnet18Rnn(params_model)
    else:
        model = models.video.r3d_18(pretrained=True, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    return model


import cv2
import numpy as np

def get_frames(filename, n_frames=1):
    frames = []
    video_capture = cv2.VideoCapture(filename)
    video_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, video_len-1, n_frames+1, dtype=np.int16)

    for frame_num in range(video_len):
        success, frame = video_capture.read()
        if success is False:
            continue
        if (frame_num in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    video_capture.release()
    return frames, video_len


import torchvision.transforms as transforms
from PIL import Image

def transform_frames(frames, model_type="rnn"):
    if model_type == "rnn":
        height, width = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    test_transformer = transforms.Compose([
                            transforms.Resize((height, width)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])

    frames_transform = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_transform = test_transformer(frame)
        frames_transform.append(frame_transform)
    images_tensor = torch.stack(frames_transform)

    if model_type=="3dcnn":
        images_tensor = torch.transpose(images_tensor, 1, 0)
    images_tensor = images_tensor.unsqueeze(0)
    return images_tensor


def store_frames(frames, path_to_store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_image = os.path.join(path_to_store, "frame" + str(ii) + ".jpg")
        cv2.imwrite(path_to_image, frame)
