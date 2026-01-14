import json


with open("longmemeval_m_cleaned.json", "r") as f:
    data = json.load(f)

total_questions = 10

dataset = []

for item in data:
    if len(dataset) >= total_questions:
        break
    question = item["question"]
    answer = item["answer"]
    session = item["haystack_sessions"]

    dataset.append({"question": question, "answer": answer, "session": session})


with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)