import streamlit as st
import pandas as pd
import numpy as np

import torch
import torchvision

from utils.process import convert_text_to_embeddings
from architecture import face, mnist

@st.cache
def read_csv(path):
    df = pd.read_csv(path).drop(columns=['Step'])
    return df

@st.cache
def load_face_generators(device):
    generator64 = face.Generator64()
    generator256 = face.Generator256()
    
    if device.type == 'cuda':
        generator64.load_state_dict(torch.load('model/face/gen64.pt'))
        generator256.load_state_dict(torch.load('model/face/gen256.pt'))
    else:
        generator64.load_state_dict(torch.load('model/face/gen64.pt', map_location=lambda storage, loc: storage))
        generator256.load_state_dict(torch.load('model/face/gen256.pt', map_location=lambda storage, loc: storage))
    
    return generator64.eval(), generator256.eval()

@st.cache
def load_mnist_generators(device):
    generator_gan = mnist.GAN()
    generator_dcgan = mnist.DCGAN()
    generator_cgan = mnist.CGAN()
    generator_acgan = mnist.ACGAN()

    if device.type == 'cuda':
        generator_gan.load_state_dict(torch.load('model/mnist/digit/gan.pt'))
        generator_dcgan.load_state_dict(torch.load('model/mnist/fashion/dcgan.pt'))
        generator_cgan.load_state_dict(torch.load('model/mnist/digit/cgan.pt'))
        generator_acgan.load_state_dict(torch.load('model/mnist/fashion/acgan.pt'))
    else:
        generator_gan.load_state_dict(torch.load('model/mnist/digit/gan.pt', map_location=lambda storage, loc: storage))
        generator_dcgan.load_state_dict(torch.load('model/mnist/fashion/dcgan.pt', map_location=lambda storage, loc: storage))
        generator_cgan.load_state_dict(torch.load('model/mnist/digit/cgan.pt', map_location=lambda storage, loc: storage))
        generator_acgan.load_state_dict(torch.load('model/mnist/fashion/acgan.pt', map_location=lambda storage, loc: storage))
    
    return generator_gan.eval(), generator_dcgan.eval(), generator_cgan.eval(), generator_acgan.eval()

@st.cache
def get_output(generator, device, shape, user_input=None, input_type='mnist'):
    input_noise = torch.randn(size=shape).to(device)

    if user_input is not None:
        if input_type == 'face':
            embeddings = convert_text_to_embeddings([user_input])
            if shape[0] > 1:
                embeddings = embeddings.repeat(shape[0], 1)
            output = generator(input_noise, embeddings)
        else:
            input_labels = torch.tensor([user_input for _ in range(shape[0])]).to(device)
            output = generator(input_noise, input_labels)
    else:
        output = generator(input_noise)

    nrow = 1
    if shape[0] > 1:
        nrow = shape[0] // 4
    norm_output = torchvision.utils.make_grid(output.detach(), nrow=nrow, normalize=True)
    return np.transpose(norm_output.numpy(), (1, 2, 0))



def main():
    st.title('Face Generation from Textual Description')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##### Face GANS #####
    st.subheader('Face Generation')

    # First load the model
    generator64, generator256 = load_face_generators(device)

    # Take in input text
    user_input = st.text_input('Enter a text description', 'He is a man.')


    # Feed to Generator
    output64 = get_output(generator64, device, (16, 100), user_input=user_input, input_type='face')    
    output256 = get_output(generator256, device, (1, 100), user_input=user_input, input_type='face')

    # Display image
    st.image(output64)
    st.image(output256)

    ##### MNIST GANS #####
    st.subheader('MNIST Models')

    # Load models
    generator_gan, generator_dcgan, generator_cgan, generator_acgan = load_mnist_generators(device)

    # GAN
    gan_df = read_csv('history/GANS.csv')
    st.line_chart(data=gan_df)

    gan_output = get_output(generator_gan, device, (64, 100))
    st.image(gan_output)

    # DCGAN
    dcgan_df = read_csv('history/DCGANS.csv')
    st.line_chart(data=dcgan_df)

    dcgan_output = get_output(generator_dcgan, device, (64, 100, 1, 1))
    st.image(dcgan_output)

    label = st.slider('Slide for different values!', min_value=0, max_value=9, value=0)

    # CGAN
    cgan_df = read_csv('history/CGANS.csv')
    st.line_chart(data=cgan_df)

    cgan_output = get_output(generator_cgan, device, (64, 100), user_input=label)
    st.image(cgan_output)

    # ACGAN
    st.text("0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'")
    acgan_df = read_csv('history/ACGANS.csv')
    st.line_chart(data=acgan_df)

    acgan_output = get_output(generator_acgan, device, (64, 100), user_input=label)
    st.image(acgan_output)

if __name__ == "__main__":
    main()