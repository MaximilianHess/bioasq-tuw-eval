import json
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import os

def create_df_from_json(path):
    with open(path, "r") as file:
        json_data = json.load(file)

    
    rows = []
    id = 0
    for question in json_data.get("questions", []):
        # Extract the body (question text) and the list of documents
        id = question.get("id")
        documents = question.get("documents", [])

        
        doc_ids = [document.split("/")[-1] for document in documents]

        rows.append({"id": id, "document_ids": doc_ids})

    df = pd.DataFrame(rows)
    df.reset_index(drop=True, inplace=True)
    return df


def recall(retrieved, relevant):
    retrieved = set(retrieved)
    retrieved_relevant = [doc for doc in retrieved if doc in relevant]
    return len(retrieved_relevant) / len(relevant)


def precision(retrieved, relevant, k=10):
    retrieved_k = retrieved[:k]
    retrieved_k = set(retrieved_k)
    return sum(1 for doc in retrieved_k if doc in relevant) / len(retrieved_k)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def average_precision(retrieved, relevant):
    relevant = list(dict.fromkeys(relevant))

    score = 0.0
    num_hits = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(len(relevant),10)

def gmap(average_precision, epsilon = 0.00001):
    gmaps = [avg_p + epsilon for avg_p in average_precision]
    return round(np.exp(np.mean(np.log(gmaps))),4)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    cfg = load_config("config.yaml")

    batch_num = cfg.get("batch", 1)
    gt_path = Path("data") / "ground_truth" / f"ground_truth_batch{batch_num}.json"
    ground_truth = create_df_from_json(gt_path)

    pred_path = cfg["pred_path"]
    pred = create_df_from_json(pred_path)
    
    recalls = []
    precisions = []
    f1_scores = []
    average_precisions = []
    for i in range(len(ground_truth)):
        # Get the ground truth and predicted documents for the current question
        gt = ground_truth.iloc[i]
        pred_row = pred[pred["id"] == gt["id"]]
        retrieved = pred_row.iloc[0]["document_ids"]
        relevant = gt["document_ids"]

        # If there are no documents retrieved, set recall and precision to 0
        if len(retrieved) == 0:
            recalls.append(0.0)
            precisions.append(0.0)
            f1_scores.append(0.0)
            average_precisions.append(0.0)
            continue

        
        p = precision(retrieved,relevant)
        precisions.append(p)
        r = recall(retrieved, relevant)
        recalls.append(r)
        f1 = f1_score(p, r)
        f1_scores.append(f1)
        avg_p = average_precision(retrieved, relevant)
        average_precisions.append(avg_p)

    # Save results
    mean_precision = round(np.mean(precisions),4)
    rec = round(np.mean(recalls),4)
    f1 = round(np.mean(f1_scores),4)
    mean_avg_precision = round(np.mean(average_precisions),4)
    gmap_score = round(gmap(average_precisions),4)

    
    # Wrte results to a JSON file
    results = {
        "mean_precision": mean_precision,
        "recall": rec,
        "f_measure": f1,
        "map": mean_avg_precision,
        "gmap": gmap_score
    }

    out_name = cfg.get("results_name")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not out_name:
        out_name = Path("results")/f"results_batch{batch_num}_{ts}.json"
    else:
        if not out_name.endswith(".json"):
            out_name = f"{out_name}_{ts}.json"
        out_name = Path("results")/out_name


    os.makedirs("results", exist_ok=True)
    
    with open(out_name, "w") as f:
        json.dump(results, f, indent=4)

        
    print("Mean Precision: ", round(np.mean(precisions),4))
    print("Recall: ", round(np.mean(recalls),4))
    print("F-Measure: ", round(np.mean(f1_scores),4))
    print("MAP: ", round(np.mean(average_precisions),4))
    print("GMAP: ", gmap(average_precisions))