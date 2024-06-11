# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
from flask_cors import CORS, cross_origin
# creating a Flask app 
app = Flask(__name__) 
cors = CORS(app)

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("model/phase7")
# model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

@app.route('/summary', methods=['POST'])
@cross_origin()
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    clean_text = remove_html_tags(text)
    # clean_text =  "vietnews: " + clean_text
    if not text:
        return jsonify({"error": "No text provided"}), 400
    encoding = tokenizer(clean_text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    # Generate the summary
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        early_stopping=True
    )
    lines = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        lines.append(line)
    summary = ". ".join(lines)
    summary = summary.replace("* ", "")
    summary = summary[0].upper() + summary[1:]
    return jsonify({"summary": summary})

@app.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"
# driver function 
if __name__ == '__main__': 
   app.run(host='0.0.0.0', port=5005, debug=True)