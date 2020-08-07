import streamlit as st
import numpy as np
import torch
import torchvision

from process import convert_text_to_embeddings
from generator_architecture import Generator

def load_generator(device):
    generator = Generator()
    if device.type == 'cuda':
        generator.load_state_dict(torch.load('model/generator.pt'))
    else:
        generator.load_state_dict(torch.load('model/generator.pt', map_location=lambda storage, loc: storage))
    return generator.eval()

def test_main():

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # First load the model
    generator = load_generator(device)

    # Take in input text
    user_input = 'He is a man.'

    # Convert to embeddings
    text_embeddings = convert_text_to_embeddings([user_input])
    print(text_embeddings.shape)

    # Feed to Generator
    input_noise = torch.randn(size=(1, 100)).to(device)
    output = generator(input_noise, text_embeddings)

    norm_output = torchvision.utils.make_grid(output.detach(), nrow=1, normalize=True)

    import matplotlib.pyplot as plt

    plt.imshow(np.transpose(norm_output, (1, 2, 0)))
    plt.show()

def main():
    st.title('Face Generation from Textual Description')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # First load the model
    generator = load_generator(device)

    # Take in input text
    user_input = st.text_input('Enter a text description', 'He is a man.')

    # Convert to embeddings
    text_embeddings = convert_text_to_embeddings([user_input])
    print(text_embeddings.shape)

    # Feed to Generator
    input_noise = torch.randn(size=(1, 100)).to(device)
    output = generator(input_noise, text_embeddings)

    norm_output = torchvision.utils.make_grid(output.detach(), nrow=1, normalize=True).numpy()

    # Display image
    st.image(np.transpose(norm_output, (1, 2, 0)))

if __name__ == "__main__":
    main()