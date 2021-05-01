import streamlit as st
import re


class Toc:
    """
    Implementation found on https://discuss.streamlit.io/t/table-of-contents-widget/3470/8
    """

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text, key):
        self._markdown(text, text, key, "h1")

    def header(self, text, toc_text, key):
        self._markdown(text, toc_text, key, "h2", " " * 2)

    def placeholder(self, sidebar=True):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, toc_text, key, level, space=""):
        key = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "-").lower()
        tags = {"h1": "bold", "h2": ""}

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)

        style = tags[level]

        self._items.append(
            f"{space}* <a style='color: #F63366; font-weight: {style}; text-decoration: none' href='#{key}'>{toc_text}</a>"
        )
