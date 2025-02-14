import os

from speech_to_text import start_recording_, stop_recording_

from vector_database import (
    add_documents_,
    add_pdf_documents_,
    get_documents_,
    get_pdf_documents_,
    query_documents_,
    query_pdf_documents_,
)

from pdf_parser import chunk_pdf_

import requests
import json

from flask import Flask, Response, request, jsonify


LLM_API_URL = "http://localhost:1234/api/v0/chat/completions"
LLM_API_HEADERS = {"Content-Type": "application/json"}

UPLOAD_FOLDER = "uploads"

user_question = ""


app = Flask(__name__)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    global user_question
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    return jsonify({"status": "Question received"}), 200


@app.route("/stream", methods=["GET"])
def stream_answer():
    global user_question
    if not user_question:
        return jsonify({"error": "No question to process"}), 400

    # Create system prompt from stored documents
    documents_query_result = query_documents_([user_question])
    documents = documents_query_result["documents"][0]
    pdf_documents_query_result = query_pdf_documents_([user_question])
    pdf_documents = pdf_documents_query_result["documents"][0]
    pdf_pages = [
        metadata["page"] for metadata in pdf_documents_query_result["metadatas"][0]
    ]
    pdf_documents = [
        f"Page {page}: {document}" for page, document in zip(pdf_pages, pdf_documents)
    ]
    system_prompt = f"""
    You are a helpful assistant that helps solve technical problems.
    If you are not sure in whatever you answer, say that you don't know.
    Answer in maximum 5 sentences.
    You have the following information from the user available.
    Refer to it as user information:{"\n".join(documents)}
    You have the following information from literature available.
    The page number is in front of the information. 
    Refer to if as literature information and always say from which page it is:{"\n".join(pdf_documents)}
    Always prefer the user information over the literature information if its relevant.
    Mention it if its relevant.
    """
    # print(system_prompt)
    # system_prompt = SYSTEM_PROMPT

    # Payload for the external API
    payload = {
        "model": "mistral-nemo-instruct-2407",
        "messages": [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": user_question},
        ],
        "temperature": 0.3,
        "max_tokens": -1,
        "stream": True,
    }

    def generate():
        with requests.post(
            LLM_API_URL, headers=LLM_API_HEADERS, json=payload, stream=True
        ) as response:
            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    try:
                        chunk = chunk[6:]  # Strip 'data: '
                        data = json.loads(chunk)
                        finish_reason = data["choices"][0]["finish_reason"]

                        if finish_reason:
                            yield "data: [DONE]\n\n"
                            break

                        content = data["choices"][0]["delta"]["content"]
                        yield f"data: {content}\n\n"

                    except json.JSONDecodeError:
                        yield f"data: {chunk}\n\n"

        yield "\n"

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.route("/start_recording", methods=["POST"])
def start_recording():
    status = start_recording_()
    if not status:
        return jsonify({"error": "Recording already running"}), 400
    return jsonify({"status": "Recording started"}), 200


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    result = stop_recording_()
    if not result:
        return jsonify({"error": "No audio recorded"}), 400
    return jsonify(result), 200


@app.route("/documents", methods=["GET"])
def get_documents():
    documents = get_documents_()
    if not documents:
        return jsonify({"error": "Documents could not be fetched"}), 400
    return jsonify(documents), 200


@app.route("/pdf_documents", methods=["GET"])
def get_pdf_documents():
    documents = get_pdf_documents_()
    if not documents:
        return jsonify({"error": "PDF documents could not be fetched"}), 400
    return jsonify(documents), 200


@app.route("/documents", methods=["POST"])
def add_documents():
    documents = request.json.get("documents")
    status = add_documents_(documents=documents)
    if not status:
        return jsonify({"error": "Documents not added"}), 400
    return jsonify({"status": "Documents added"}), 200


@app.route("/pdf_documents", methods=["POST"])
def add_pdf_documents():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    pdf_file = request.files["file"]
    if pdf_file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(path)
    chapter_pages, chapter_page_numbers, chapter_titles = chunk_pdf_(path)
    document_title = os.path.splitext(os.path.basename(path))[0]
    status = add_pdf_documents_(
        chapter_pages,
        chapter_page_numbers,
        chapter_titles,
        document_title=document_title,
    )
    if not status:
        return jsonify({"error": "Documents not added"}), 400
    return jsonify({"status": "Documents added"}), 200


@app.route("/documents/query", methods=["GET"])
def query_documents():
    queries = documents = request.json.get("queries")
    documents = query_documents_(queries)
    if not documents:
        return jsonify({"error": "Document could not be queried"}), 400
    return jsonify(documents), 200


@app.route("/pdf_documents/query", methods=["GET"])
def query_pdf_documents():
    queries = documents = request.json.get("queries")
    documents = query_pdf_documents_(queries)
    if not documents:
        return jsonify({"error": "Documents could not be queried"}), 400
    return jsonify(documents), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
