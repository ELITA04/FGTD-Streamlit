import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64

import torch
import torchvision
from torchvision.utils import save_image

from utils.process import convert_text_to_embeddings
from utils.toc import Toc
from architecture import face, mnist


@st.cache
def read_csv(path):
    df = pd.read_csv(path).drop(columns=["Step"])
    return df


@st.cache
def load_face_generators(device):
    dcgan64 = face.DCGAN64()
    dcgan256 = face.DCGAN256()
    sagan128 = face.SAGAN()
    dfgan128 = face.DFGAN()

    dcgan64.load_state_dict(torch.load("model/face/dcgan64.pt", map_location=device))
    dcgan256.load_state_dict(torch.load("model/face/dcgan256.pt", map_location=device))
    sagan128.load_state_dict(torch.load("model/face/sagan128.pt", map_location=device))
    dfgan128.load_state_dict(torch.load("model/face/dfgan128.pt", map_location=device))

    return dcgan64.eval(), dcgan256.eval(), sagan128.eval(), dfgan128.eval()


@st.cache
def load_mnist_generators(device):
    gan = mnist.GAN()
    dcgan = mnist.DCGAN()
    cgan = mnist.CGAN()
    acgan = mnist.ACGAN()

    gan.load_state_dict(torch.load("model/mnist/digit/gan.pt", map_location=device))
    dcgan.load_state_dict(
        torch.load("model/mnist/fashion/dcgan.pt", map_location=device)
    )
    cgan.load_state_dict(torch.load("model/mnist/digit/cgan.pt", map_location=device))
    acgan.load_state_dict(
        torch.load("model/mnist/fashion/acgan.pt", map_location=device)
    )

    return gan.eval(), dcgan.eval(), cgan.eval(), acgan.eval()


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


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def main():

    toc = Toc()
    toc.header("Face Generation from Textual Description  👩 👨 📋 ")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Face GANS #####
    st.markdown("### Demo  ")
    st.markdown("Image size : 64x64  and 256 x 256")

    # First load the model
    dcgan64, dcgan256, sagan128, dfgan128 = load_face_generators(device)

    # Take in input text
    # Examples Button
    if st.button("Example 1"):
        user_input = "The woman has high cheekbones. She has straight hair which is black in colour. She has big lips with arched eyebrows. The smiling, young woman has rosy cheeks and heavy makeup. She is wearing lipstick."

    if st.button("Example 2"):
        user_input = "The man sports a 5 o’clock shadow and mustache. He has a receding hairline. He has big lips and big nose, narrow eyes and a slightly open mouth. The young attractive man is smiling. He’s wearing necktie."

    if st.button("Example 3"):
        user_input = "The man has straight hair. He has arched eyebrows.The man looks young and attractive. He’s wearing necktie."

    try:
        user_input = st.text_area("Try it yourself!", user_input)
    except NameError:
        user_input = st.text_area(
            "Try it yourself!",
            "The man sports a 5 o’clock shadow. His hair is black in colour. He has big nose with bushy and arched eyebrows. The man looks attractive.",
        )

    # Feed to Generator
    output64 = get_output(
        dcgan64, device, (1, 100), user_input=user_input, input_type="face"
    )
    output256 = get_output(
        dcgan256, device, (1, 100), user_input=user_input, input_type="face"
    )
    output_sagan = get_output(
        sagan128, device, (1, 100), user_input=user_input, input_type="face"
    )
    output_dfgan = get_output(
        dfgan128, device, (1, 100), user_input=user_input, input_type="face"
    )

    # Display image

    st.image([output64, output256, output_sagan, output_dfgan])

    st.markdown("---")

    ##### MNIST GANS #####
    toc.header("MNIST Dataset (Digit and Fashion) ")

    # Load models
    gan, dcgan, cgan, acgan = load_mnist_generators(device)

    toc.subheader("GANs ")
    st.markdown("Epochs : 20 ")
    # GAN
    gan_df = read_csv("history/gan.csv")
    st.line_chart(data=gan_df)

    gan_output = get_output(gan, device, (64, 100))
    st.image(gan_output)
    st.markdown("---")

    # DCGAN
    toc.subheader("Deep Convolution GANs ")
    st.markdown("Epochs : 20 ")
    dcgan_df = read_csv("history/dcgan.csv")
    st.line_chart(data=dcgan_df)

    dcgan_output = get_output(dcgan, device, (64, 100, 1, 1))
    st.image(dcgan_output)
    st.markdown("---")

    # CGAN
    toc.subheader("Conditional GANs ")
    st.markdown("Epochs : 20 ")

    cgan_label = st.slider(
        "Slide for different digit images!", min_value=0, max_value=9, value=0
    )

    cgan_output = get_output(cgan, device, (64, 100), user_input=cgan_label)
    st.image(cgan_output)
    cgan_df = read_csv("history/cgan.csv")
    st.line_chart(data=cgan_df)
    st.markdown("---")

    # ACGAN
    toc.subheader("Auxilary Conditional GANs ")
    st.markdown("Epochs : 20 ")

    acgan_label = st.slider(
        "Slide for different fashion images!", min_value=0, max_value=9, value=0
    )
    st.text(
        "0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'"
    )

    acgan_output = get_output(acgan, device, (64, 100), user_input=acgan_label)
    st.image(acgan_output)
    acgan_df = read_csv("history/acgan.csv")
    st.line_chart(data=acgan_df)
    st.markdown("---")

    toc.placeholder()
    toc.generate()


if __name__ == "__main__":
    main()
