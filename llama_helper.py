from ollama import chat

from faiss_helper import vectorstore
retriever = vectorstore.as_retriever()


query = "What are the technical skills mentioned in the resume?"

docs = retriever.get_relevant_documents(query)
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

if not docs:
    print("No relevant documents found.")

prompt ="""You are required to answer the question based on the context provided.
and if there is no answer in the context, say 'I don't know'.
Context:
{context}

Question:
{query}


"""

result = chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": prompt.format(context="\n".join([doc.page_content for doc in docs]), query=query)}
    ],
    stream=False, 
    )
print("Response:", result.message.content)