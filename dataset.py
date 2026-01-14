import json


with open("longmemeval_m_cleaned.json", "r") as f:
    data = json.load(f)

total_questions = 10

dataset = []

for item in data:
    if len(dataset) >= total_questions:
        break
    question = item["question"]
    session = item["haystack_sessions"]
    dataset.append({"question": question, "session": session})


with open("dataset.json", "w") as f:
    json.dump(dataset, f)