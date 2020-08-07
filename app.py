import streamlit as st
import numpy as np
import torch
import torchvision

from utils.process import convert_text_to_embeddings
from architecture import gen64, gen256

def load_generators(device):
    generator64 = gen64.Generator()
    generator256 = gen256.Generator()
    if device.type == 'cuda':
        generator64.load_state_dict(torch.load('model/gen64.pt'))
        generator256.load_state_dict(torch.load('model/gen256.pt'))
    else:
        generator64.load_state_dict(torch.load('model/gen64.pt', map_location=lambda storage, loc: storage))
        generator256.load_state_dict(torch.load('model/gen256.pt', map_location=lambda storage, loc: storage))
    return generator64.eval(), generator256.eval()

def test_main():

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # First load the model
    generator64, generator256 = load_generators(device)

    # Take in input text
    user_input = 'He is a man.'

    # Convert to embeddings
    text_embeddings = convert_text_to_embeddings([user_input])

    # Feed to Generator
    input_noise = torch.randn(size=(1, 100)).to(device)
    output64 = generator64(input_noise, text_embeddings)
    output256 = generator256(input_noise, text_embeddings)

    norm_output64 = torchvision.utils.make_grid(output64.detach(), nrow=1, normalize=True)
    norm_output256 = torchvision.utils.make_grid(output256.detach(), nrow=1, normalize=True)


    import matplotlib.pyplot as plt

    plt.imshow(np.transpose(norm_output64, (1, 2, 0)))
    plt.imshow(np.transpose(norm_output256, (1, 2, 0)))
    plt.show()

def main():
    st.title('Face Generation from Textual Description')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # First load the model
    generator64, generator256 = load_generators(device)

    # Take in input text
    user_input = st.text_input('Enter a text description', 'He is a man.')

    # Convert to embeddings
    text_embeddings = convert_text_to_embeddings([user_input])

    # Feed to Generator
    input_noise = torch.randn(size=(1, 100)).to(device)
    output64 = generator64(input_noise, text_embeddings)
    output256 = generator256(input_noise, text_embeddings)

    norm_output64 = torchvision.utils.make_grid(output64.detach(), nrow=1, normalize=True)
    norm_output256 = torchvision.utils.make_grid(output256.detach(), nrow=1, normalize=True)

    # Display image
    st.image(np.transpose(norm_output64.numpy(), (1, 2, 0)))
    st.image(np.transpose(norm_output256.numpy(), (1, 2, 0)))

if __name__ == "__main__":
    main()