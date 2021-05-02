import streamlit as st
import torch

from architecture import face, mnist


@st.cache
def load_face_generators(device):
    """
    Loads all the Face GAN models.

    Arguments:
        device: Current backend device.
    """
    dcgan = face.DCGAN()
    n_dcgan = face.DCGAN()

    sagan = face.SAGAN()
    n_sagan = face.SAGAN()

    dfgan = face.DFGAN()
    n_dfgan = face.DFGAN()

    dcgan.load_state_dict(torch.load("models/face/dcgan/dcgan.pt", map_location=device))
    n_dcgan.load_state_dict(
        torch.load("models/face/dcgan/n_dcgan.pt", map_location=device)
    )

    sagan.load_state_dict(torch.load("models/face/sagan/sagan.pt", map_location=device))
    n_sagan.load_state_dict(
        torch.load("models/face/sagan/n_sagan.pt", map_location=device)
    )

    dfgan.load_state_dict(torch.load("models/face/dfgan/dfgan.pt", map_location=device))
    n_dfgan.load_state_dict(
        torch.load("models/face/dfgan/n_dfgan.pt", map_location=device)
    )

    return (
        dcgan.eval(),
        n_dcgan.eval(),
        sagan.eval(),
        n_sagan.eval(),
        dfgan.eval(),
        n_dfgan.eval(),
    )


@st.cache
def load_mnist_generators(device):
    """
    Loads all the MNIST GAN models.

    Arguments:
        device: Current backend device.
    """
    gan = mnist.GAN()
    dcgan = mnist.DCGAN()
    cgan = mnist.CGAN()
    acgan = mnist.ACGAN()

    gan.load_state_dict(torch.load("models/mnist/digit/gan.pt", map_location=device))
    dcgan.load_state_dict(
        torch.load("models/mnist/fashion/dcgan.pt", map_location=device)
    )
    cgan.load_state_dict(torch.load("models/mnist/digit/cgan.pt", map_location=device))
    acgan.load_state_dict(
        torch.load("models/mnist/fashion/acgan.pt", map_location=device)
    )

    return gan.eval(), dcgan.eval(), cgan.eval(), acgan.eval()
