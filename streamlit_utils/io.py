import streamlit as st
import pandas as pd
import numpy as np

import torch
import torchvision
from torchvision.utils import save_image

from utils.process import convert_text_to_embeddings

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_sample(df):
    """
    Get a sample text from the seen or unseen files.

    Arguments:
        df: Dataframe containing texts.
    """
    sample = df.sample(replace=True)
    return sample["text_description"].values[0]


@st.cache
def read_csv(path):
    """
    Read csv from path provided.

    Arguments:
        path: Path of the csv.
    """
    df = pd.read_csv(path)
    return df


@st.cache
def get_output(generator, device, shape, user_input=None, input_type="mnist"):
    """
    Generates outputs from the model and converts them into a gridded image.

    Arguments:
        generator: Generator model.
        device: Current backend device.
        shape: Shape of the noise vector.
        user_input: Conditionally generate based on user input.
        input_type: Whether the model is an MNIST GAN model or Face GAN model.
    """
    input_noise = torch.randn(size=shape).to(device)

    divs = 4 if input_type == "mnist" else 2

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
        nrow = shape[0] // divs
    norm_output = torchvision.utils.make_grid(
        output.detach(), nrow=nrow, normalize=True
    )

    return np.transpose(norm_output.numpy(), (1, 2, 0))


def face_graph(df, n_df, subplot_titles):
    """
    Produces a Loss vs Epoch graph of the respective Face GAN model.

    Arguments:
        df: Dataframe for single caption model.
        n_df: Dataframe for N-caption model.
        subplot_titles: Names of the single caption and N-caption model respectively.
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=subplot_titles)

    step = df["step"].to_list()
    discriminator_loss = df["discriminator"].to_list()
    generator_loss = df["generator"].to_list()
    fig.add_trace(
        go.Scatter(
            x=step,
            y=generator_loss,
            mode="lines+markers",
            name="Generator Loss",
            line_color="#f2433a",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=step,
            y=discriminator_loss,
            mode="lines+markers",
            name="Discriminator Loss",
            line_color="#3f60cc",
        ),
        row=1,
        col=1,
    )

    step = n_df["step"].to_list()
    discriminator_loss = n_df["discriminator"].to_list()
    generator_loss = n_df["generator"].to_list()
    fig.add_trace(
        go.Scatter(
            x=step,
            y=generator_loss,
            mode="lines+markers",
            name="Generator Loss",
            line_color="#f2433a",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=step,
            y=discriminator_loss,
            mode="lines+markers",
            name="Discriminator Loss",
            line_color="#3f60cc",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Epoch", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Epoch", showgrid=False, row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    # Update size
    fig.update_layout(height=600, title_text="Loss vs Epoch")

    st.plotly_chart(fig, use_container_width=True)


def mnist_graph(df):
    """
    Produces a Loss vs Epoch graph of the respective MNIST GAN model.

    Arguments:
        df: Dataframe for the model.
    """

    fig = go.Figure()

    step = df["step"].to_list()
    discriminator_loss = df["discriminator"].to_list()
    generator_loss = df["generator"].to_list()
    fig.add_trace(
        go.Scatter(
            x=step,
            y=generator_loss,
            mode="lines+markers",
            name="Generator Loss",
            line_color="#f2433a",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step,
            y=discriminator_loss,
            mode="lines+markers",
            name="Discriminator Loss",
            line_color="#3f60cc",
        )
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Epoch", showgrid=False)

    # Update yaxis properties
    fig.update_yaxes(title_text="Loss")

    # Update size
    fig.update_layout(height=300, title_text="Loss vs Epoch")

    st.plotly_chart(fig, use_container_width=True)
