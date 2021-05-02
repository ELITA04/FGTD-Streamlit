import streamlit as st
import torch
import random
import gc

from utils.toc import Toc
from utils.model_downloader import download_models
from utils.footer import footer

from streamlit_utils.loaders import load_face_generators, load_mnist_generators
from streamlit_utils.io import get_sample, read_csv, get_output, face_graph, mnist_graph


def main():

    gc.enable()
    ##### Setup #####
    toc = Toc()
    toc.title("Face Generation from Textual Description  👩 👨 📋 ", "fgtd")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Face GANS #####
    st.markdown("---")
    st.markdown("## Demo")

    # First load the model
    dcgan, n_dcgan, sagan, n_sagan, dfgan, n_dfgan = load_face_generators(device)

    ##### Examples #####
    seen_df = read_csv("examples/seen_text.csv")
    unseen_df = read_csv("examples/unseen_text.csv")

    butt_col1, butt_col2, butt_col3, butt_col4, butt_col5 = st.beta_columns(5)

    with butt_col1:
        if st.button("Example 1"):
            user_input = "The woman has high cheekbones. She has straight hair which is black in colour. She has big lips with arched eyebrows. The smiling, young woman has rosy cheeks and heavy makeup. She is wearing lipstick."

    with butt_col2:
        if st.button("Example 2"):
            user_input = "The man sports a 5 o’clock shadow and mustache. He has a receding hairline. He has big lips and big nose, narrow eyes and a slightly open mouth. The young attractive man is smiling. He’s wearing necktie."

    with butt_col3:
        if st.button("Example 3"):
            user_input = "The man has straight hair. He has arched eyebrows. The man looks young and attractive. He’s wearing necktie."

    with butt_col4:
        if st.button("Trained", help="Get a random example from training set."):
            user_input = get_sample(seen_df)

    with butt_col5:
        if st.button("Unseen", help="Get a random example from untrained text."):
            user_input = get_sample(unseen_df)

    try:
        user_input = st.text_area("Try it yourself!", user_input)
    except NameError:
        user_input = st.text_area(
            "Try it yourself!",
            "The man sports a 5 o’clock shadow. His hair is black in colour. He has big nose with bushy and arched eyebrows. The man looks attractive.",
        )

    st.markdown("---")

    ##### Outputs #####

    ##### DCGAN #####
    toc.header("Deep Convolution GAN", "DCGAN", "face-dcgan")
    output_dcgan = get_output(
        dcgan, device, (4, 100), user_input=user_input, input_type="face"
    )
    output_ndcgan = get_output(
        n_dcgan, device, (4, 100), user_input=user_input, input_type="face"
    )

    dc_col1, dc_col2 = st.beta_columns(2)

    with dc_col1:
        st.image(
            [
                output_dcgan,
            ],
            caption=["DCGAN"],
        )

    with dc_col2:
        st.image(
            [
                output_ndcgan,
            ],
            caption=["N DCGAN"],
        )

    dcgan_df = read_csv("history/face/dcgan.csv")
    n_dcgan_df = read_csv("history/face/n-dcgan.csv")
    face_graph(dcgan_df, n_dcgan_df, ["DCGAN", "N DCGAN"])
    st.markdown("---")

    ##### SAGAN #####
    toc.header("Self-Attention GAN", "SAGAN", "sagan")
    output_sagan = get_output(
        sagan, device, (4, 100), user_input=user_input, input_type="face"
    )
    output_nsagan = get_output(
        n_sagan, device, (4, 100), user_input=user_input, input_type="face"
    )

    sa_col1, sa_col2 = st.beta_columns(2)

    with sa_col1:
        st.image(
            [
                output_sagan,
            ],
            caption=["SAGAN"],
        )
    with sa_col2:
        st.image(
            [
                output_nsagan,
            ],
            caption=["N SAGAN"],
        )

    sagan_df = read_csv("history/face/sagan.csv")
    n_sagan_df = read_csv("history/face/n-sagan.csv")
    face_graph(sagan_df, n_sagan_df, ["SAGAN", "N SAGAN"])
    st.markdown("---")

    ##### DFGAN #####
    toc.header("Deep Fusion GAN", "DFGAN", "dfgan")
    output_dfgan = get_output(
        dfgan, device, (4, 100), user_input=user_input, input_type="face"
    )
    output_ndfgan = get_output(
        n_dfgan, device, (4, 100), user_input=user_input, input_type="face"
    )

    df_col1, df_col2 = st.beta_columns(2)

    with df_col1:
        st.image(
            [
                output_dfgan,
            ],
            caption=["DFGAN"],
        )
    with df_col2:
        st.image(
            [
                output_ndfgan,
            ],
            caption=["N DFGAN"],
        )

    dfgan_df = read_csv("history/face/dfgan.csv")
    n_dfgan_df = read_csv("history/face/n-dfgan.csv")
    face_graph(dfgan_df, n_dfgan_df, ["DFGAN", "N DFGAN"])
    st.markdown("---")

    ##### MNIST GANS #####
    toc.title("MNIST Dataset (Digit and Fashion)", "mnist")

    # Load models
    gan, dcgan, cgan, acgan = load_mnist_generators(device)

    ##### GAN #####
    toc.header("Vanilla GAN", "GAN", "gan")
    st.markdown("Epochs : 20 ")

    gan_output = get_output(gan, device, (64, 100))
    st.image(gan_output)

    gan_df = read_csv("history/mnist/gan.csv")
    mnist_graph(gan_df)
    st.markdown("---")

    ##### DCGAN #####
    toc.header("Deep Convolution GAN (MNIST)", "DCGAN", "mnist-dcgan")
    st.markdown("Epochs : 20 ")

    dcgan_output = get_output(dcgan, device, (64, 100, 1, 1))
    st.image(dcgan_output)

    dcgan_df = read_csv("history/mnist/dcgan.csv")
    mnist_graph(dcgan_df)
    st.markdown("---")

    ##### CGAN #####
    toc.header("Conditional GAN", "CGAN", "cgan")
    st.markdown("Epochs : 20 ")

    cgan_label = st.slider(
        "Slide for different digit images!", min_value=0, max_value=9, value=0
    )
    cgan_output = get_output(cgan, device, (64, 100), user_input=cgan_label)
    st.image(cgan_output)

    cgan_df = read_csv("history/mnist/cgan.csv")
    mnist_graph(cgan_df)
    st.markdown("---")

    ##### ACGAN #####
    toc.header("Auxilary Conditional GAN", "ACGAN", "acgan")
    st.markdown("Epochs : 20 ")

    acgan_label = st.slider(
        "Slide for different fashion images!", min_value=0, max_value=9, value=0
    )
    st.text(
        "0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'"
    )
    acgan_output = get_output(acgan, device, (64, 100), user_input=acgan_label)
    st.image(acgan_output)

    acgan_df = read_csv("history/mnist/acgan.csv")
    mnist_graph(acgan_df)
    st.markdown("---")

    ##### TOC #####
    toc.placeholder()
    toc.generate()

    ##### Footer #####
    footer()

    gc.collect()


if __name__ == "__main__":
    download_models()
    main()
