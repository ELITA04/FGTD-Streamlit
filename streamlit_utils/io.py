import streamlit as st
import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision.utils import save_image

from utils.process import convert_text_to_embeddings

@st.cache
def read_csv(path):
    df = pd.read_csv(path).drop(columns=["Step"])
    return df


@st.cache
def get_output(generator, device, shape, user_input=None, input_type="mnist"):
    input_noise = torch.randn(size=shape).to(device)

    if user_input is not None:
        if input_type == "face":
            embeddings = convert_text_to_embeddings([user_input])
            if shape[0] > 1:
                embeddings = embeddings.repeat(shape[0], 1)
            output = generator(input_noise, embeddings)
        else:
            input_labels = torch.tensor([user_input for _ in range(shape[0])]).to(
                device
            )
            output = generator(input_noise, input_labels)
    else:
        output = generator(input_noise)

    nrow = 1
    if shape[0] > 1:
        nrow = shape[0] // 4
    norm_output = torchvision.utils.make_grid(
        output.detach(), nrow=nrow, normalize=True
    )

    return np.transpose(norm_output.numpy(), (1, 2, 0))
