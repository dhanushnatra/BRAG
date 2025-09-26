from RAG.llama_helper import get_response

if __name__ == "__main__":
    query = "Explain the steps to set up Flutter environment?"
    response = get_response(query)
    print(response.split("\n"))