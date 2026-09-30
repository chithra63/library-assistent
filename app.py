import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=API_KEY)

with open("chatbot_config.txt", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read().strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    if len(message) > 4000:
        return jsonify({"error": "Please keep your question under 4000 characters."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1200,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            return jsonify({"error": "I could not generate a response. Please try again."}), 502

        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "Something went wrong while contacting the library assistant. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
