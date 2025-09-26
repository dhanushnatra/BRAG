from ollama import chat
from .faiss_helper import FastTextEmbeddings,ft_model
from langchain_community.vectorstores import FAISS
from .md_helper import strip_markdown 

embeddings = FastTextEmbeddings(ft_model)
vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt ="""You are required to answer the question based on the context provided.
and if there is no answer in the context, say 'I don't know'.
Context:
{context}

Question:
{query}


Answer:
format your answer in List format if it contains multiple points.

"""
def get_response(query:str)->str:

    docs = retriever.get_relevant_documents(query)
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

    if not docs:
        print("No relevant documents found.")


    result = chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt.format(context="\n".join([doc.page_content for doc in docs]), query=query)}
        ],
        stream=False,
        keep_alive=True
        )
    return str(strip_markdown(result.message.content))
