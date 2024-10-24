from flask import Flask, request, jsonify
from marker.convert import convert_single_pdf
from marker.models import load_all_models

# Create Flask app
app = Flask(__name__)

# Load the model once at the start
pdf_convert_model = load_all_models()


@app.route("/convert_pdf", methods=["POST"])
def convert_pdf():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf_file"]
    file_path = f"/tmp/{pdf_file.filename}"
    pdf_file.save(file_path)

    # Convert PDF using the preloaded model
    full_text, images, out_metadata = convert_single_pdf(file_path, pdf_convert_model)
    return jsonify({"full_text": full_text, "metadata": out_metadata})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
