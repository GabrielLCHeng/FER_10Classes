# from fastai.vision.all import *
from fastai.vision.learner import *
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from fastai.callback.hook import num_features_model
import timm
import os

import data

import pathlib

temp = pathlib.PosixPath        # reserve for linux
# some system in windows
pathlib.PosixPath = pathlib.WindowsPath


# model_inf = load_learner('./media/models/export_resnet34_customHead__acc62_42.pkl')
# img_path = './media/test_images/angry_test_00034.jpg'
# st.image(img_path)

def predict_from_frame(chosen_model:None, frame):
    if chosen_model is None:
        chosen_model = './media/models/export_resnet34__acc62_42.pkl'
    model_inf = load_learner(chosen_model)
    # make sure frame is an image
    pred_decoded, pred_class_index, probs = model_inf.predict(frame)
    prob = probs[probs.argmax(dim=0)].item()
    return pred_decoded, prob


def create_custom_head(model, n_out=None, lin_ftrs=None):
  # new_head = create_head(nf=512, n_out=10, lin_ftrs=[512, 512])   # so it becomes 2 linear layers of [nf, 512] nodes in the head, lin_ftrs must have 2 values
  # model[-1] = new_head
  nf = num_features_model(model[:-1])
  new_head = create_head(nf=nf, n_out=n_out, lin_ftrs=lin_ftrs)
  model[-1] = new_head
  return model