from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence_model = SentenceTransformer("bert-base-nli-mean-tokens").to(device)


def convert_text_to_embeddings(batch_text):
    """
    Converts the input text into embeddings.

    Arguments:
        batch_text: A list of input texts.
    """

    stack = []
    for sent in batch_text:
        l = sent.split(". ")
        sentence_embeddings = sentence_model.encode(l)
        sentence_emb = torch.FloatTensor(sentence_embeddings).to(device)
        sent_mean = torch.mean(sentence_emb, dim=0).reshape(1, 768)
        stack.append(sent_mean)
    output = torch.cat(stack, dim=0)
    return output.detach()
