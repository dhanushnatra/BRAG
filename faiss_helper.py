
# Use fasttext for embeddings
import fasttext
import fasttext.util

# Save sentences to a text file for fasttext training
sentences = [
    "the cat sat on the mat",
    "dogs are great pets",
    "I love playing football"
]
with open("sentences.txt", "w") as f:
    for s in sentences:
        f.write(s + "\n")

# Train fasttext model
ft_model = fasttext.train_unsupervised("sentences.txt", model='skipgram',minCount=1)

from langchain_community.document_loaders import PyPDFLoader
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

from langchain.embeddings.base import Embeddings
import numpy as np


class FastTextEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_query(self, text: str):
        words = text.lower().split()
        if not words:
            return np.zeros(self.model.get_dimension()).tolist()
        return np.mean([self.model.get_word_vector(w) for w in words], axis=0).tolist()

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]




from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

docs = load_pdf("data/dhanush-natra ( Resume ) .pdf")
print(f"Loaded {len(docs)} documents from PDF.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)


from langchain_community.vectorstores import FAISS

embeddings = FastTextEmbeddings(ft_model)
vectorstore = FAISS.from_documents(docs, embeddings)
