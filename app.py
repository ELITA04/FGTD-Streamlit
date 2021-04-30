import streamlit as st
import torch

from utils.toc import Toc
from utils.model_downloader import download_models

from streamlit_utils.loaders import load_face_generators, load_mnist_generators
from streamlit_utils.io import read_csv, get_output


def main():

    toc = Toc()

    toc.title("Face Generation from Textual Description  👩 👨 📋 ")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Face GANS #####
    st.markdown("### Demo")

    # First load the model
    dcgan, n_dcgan, sagan, n_sagan, dfgan, n_dfgan = load_face_generators(device)

    # Take in input text
    # Examples Button
    if st.button("Example 1"):
        user_input = "The woman has high cheekbones. She has straight hair which is black in colour. She has big lips with arched eyebrows. The smiling, young woman has rosy cheeks and heavy makeup. She is wearing lipstick."

    if st.button("Example 2"):
        user_input = "The man sports a 5 o’clock shadow and mustache. He has a receding hairline. He has big lips and big nose, narrow eyes and a slightly open mouth. The young attractive man is smiling. He’s wearing necktie."

    if st.button("Example 3"):
        user_input = "The man has straight hair. He has arched eyebrows. The man looks young and attractive. He’s wearing necktie."

    try:
        user_input = st.text_area("Try it yourself!", user_input)
    except NameError:
        user_input = st.text_area(
            "Try it yourself!",
            "The man sports a 5 o’clock shadow. His hair is black in colour. He has big nose with bushy and arched eyebrows. The man looks attractive.",
        )

    # Feed to Generator
    output_dcgan = get_output(
        dcgan, device, (1, 100), user_input=user_input, input_type="face"
    )
    output_ndcgan = get_output(
        n_dcgan, device, (1, 100), user_input=user_input, input_type="face"
    )
    
    output_sagan = get_output(
        sagan, device, (1, 100), user_input=user_input, input_type="face"
    )
    output_nsagan = get_output(
        n_sagan, device, (1, 100), user_input=user_input, input_type="face"
    )

    output_dfgan = get_output(
        dfgan, device, (1, 100), user_input=user_input, input_type="face"
    )
    output_ndfgan = get_output(
        n_dfgan, device, (1, 100), user_input=user_input, input_type="face"
    )

    # Display image

    st.image(
        [output_dcgan, output_ndcgan, output_sagan, output_nsagan, output_dfgan, output_ndfgan],
        caption=["DCGAN", "N DCGAN", "SAGAN", "N SAGAN", "DFGAN", "N DFGAN"],
    )

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
    download_models()
    main()
