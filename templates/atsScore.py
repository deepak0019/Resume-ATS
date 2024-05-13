from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from unidecode import unidecode
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

app = Flask(__name__)
tags_vals = [
    "O",
    "Date",
    "Languages",
    "Projects",
    "Hobbies",
    "Nationality",
    "Date of birth",
    "Objective",
    "LinkedIn",
    "Career Goals",
    "Profile",
    "Greeting",
    "Greeting",
    "Company Name",
    "Job Title",
    "Phone No.",
    "Website",
    "Name",
    "Address",
    "Email",
    "Companies worked at",
    "Phone No",
    "Skills",
    "Email Address",
    "Location",
    "Degree",
    "Experience",
    "College Name",
    "Graduation Year",
    "Education",
    "Designation",
    "Years of Experience",
]

tag2idx = {t: i for i, t in enumerate(tags_vals)}
idx2tag = {i: t for i, t in enumerate(tags_vals)}

MODEL_NAME = "bert-base-uncased"
TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME, lowercase=True)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2idx))
checkpoint_path = "bert-base-uncased.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def preprocess_single_resume(resume_content, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        resume_content, max_length=max_len, return_offsets_mapping=True
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    token_type_ids = tok["token_type_ids"]

    # Padding
    padding_length = max_len - len(input_ids)
    input_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    token_type_ids += [0] * padding_length

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
    }


def get_predicted_labels(model_output, idx2tag):
    _, predicted_labels = torch.max(model_output, dim=2)
    predicted_labels = predicted_labels.squeeze(0).tolist()
    predicted_labels = [idx2tag[label_idx] for label_idx in predicted_labels]
    return predicted_labels


def predict_resume_labels(resume_content, model, tokenizer, idx2tag, max_len):
    input_data = preprocess_single_resume(resume_content, tokenizer, max_len)
    input_ids = input_data["input_ids"]
    token_type_ids = input_data["token_type_ids"]
    attention_mask = input_data["attention_mask"]

    with torch.no_grad():
        model_output = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

    predicted_labels = get_predicted_labels(model_output.logits, idx2tag)
    return predicted_labels


def extract_text_from_pdf(file_path):
    output = []
    with fitz.open(file_path) as doc:
        for page in doc:
            output += page.get_text("blocks")

    clean_output = []
    for i in range(0, len(output)):
        plain_text = unidecode(output[i][4])
        clean_output.append(plain_text)
    return clean_output


def conversion(predicted_labels, tokenized_resume):
    entities = []
    current_entity = None
    start_position = None

    for idx, label in enumerate(predicted_labels):
        if label == "O":
            if current_entity is not None:
                entities.append((current_entity, start_position, idx - 1))
                current_entity = None
                start_position = None
        else:
            if "-" in label:
                entity_type = label.split("-")[1]
            else:
                entity_type = label
            if current_entity is not None and entity_type == current_entity:
                continue  # Continue with the current entity type
            elif current_entity is not None:
                entities.append((current_entity, start_position, idx - 1))
            current_entity = entity_type
            start_position = idx

    if current_entity is not None:
        entities.append((current_entity, start_position, len(predicted_labels) - 1))

    data = []
    for entity, start, end in entities:
        entity_tokens = tokenized_resume["input_ids"][start : end + 1]
        entity_text = TOKENIZER.decode(entity_tokens, skip_special_tokens=True)
        data.append({"entity": entity, "text": entity_text})

    return data


def calculate_score(entities, job_role, skills):
    # Extract provided skills from the request and convert to lowercase
    required_skills = [skill.strip().lower() for skill in skills.split(",")]

    # Initialize a set to collect required skills for the specified job role
    provided_skills = set()

    # Iterate through the entities to find job role specific skills
    for entity in entities:
        if entity["entity"] == "Skills":
            provided_skills.update(
                [skill.strip().lower() for skill in entity["text"].split()]
            )

    # Calculate the intersection (matching skills) between provided and required skills
    matched_skills = set(required_skills).intersection(provided_skills)
    num_required_skills = len(required_skills)

    # Calculate the score as the percentage of matched skills out of total required skills
    score = (
        (len(matched_skills) / num_required_skills) * 100
        if num_required_skills > 0
        else 0
    )

    return score


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_resume():

    if request.method == "POST":
        f = request.files["file"]
        f.save(f.filename)

    job_role = request.form["job-role"]
    skills = request.form["skills"]
    file_path = f.filename
    resume_text = extract_text_from_pdf(file_path)
    final_resume_text = "".join(resume_text)  # Concatenate all text blocks
    tokenized_resume = TOKENIZER.encode_plus(
        final_resume_text, max_length=500, return_offsets_mapping=True
    )
    predicted_labels = predict_resume_labels(
        final_resume_text, model, TOKENIZER, idx2tag, max_len=500
    )
    extracted_entities = conversion(predicted_labels, tokenized_resume)
    job_role = "Full Stack Developer"
    score = calculate_score(extracted_entities, job_role, skills)

    response = {"entities": extracted_entities, "score": score}

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=9999)
