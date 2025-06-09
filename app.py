from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

# 🔐 本番用：APIキーはRenderの環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")

# ChromaベクトルDB読み込み
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# QAチェーン構築
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=vectorstore.as_retriever()
)

# エンドポイント定義
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "質問が空です"}), 400

    try:
        answer = qa_chain.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 起動（Renderが自動で起動してくれる）
if __name__ == "__main__":
    app.run()