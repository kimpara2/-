from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

loader = PyPDFLoader("恋愛心理学の基礎（日本語対応）.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embedding)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")

    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(user_question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
あなたは恋愛カウンセラーです。
以下の資料に基づいて、ユーザーの相談に対して優しく的確に答えてください。

資料：
{context}

相談内容：
{user_question}
"""
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    response = model.predict(prompt)

    return jsonify({"answer": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Renderが自動で設定するPORT番号
    app.run(host="0.0.0.0", port=port)