# File: /sanchi_nlp/emotion_engine/load_and_preprocess.py

import pandas as pd

def load_goemotions_dataset(data_folder='data'):
    # Load emotion names
    with open(f"{data_folder}/emotions.txt", "r") as f:
        emotion_list = [line.strip() for line in f.readlines()]

    # Convert label string like "5,6" to ['joy', 'surprise']
    def map_labels(label_str):
        labels = list(map(int, label_str.split(',')))
        return [emotion_list[i] for i in labels]

    # Load individual .tsv file
    def load_file(path):
        df = pd.read_csv(path, sep='\t', header=None, names=["text", "labels", "ids"])
        df["emotions"] = df["labels"].apply(map_labels)
        return df[["text", "emotions"]]

    print("🔄 Loading GoEmotions dataset...")
    train_df = load_file(f"{data_folder}/train.tsv")
    dev_df = load_file(f"{data_folder}/dev.tsv")
    test_df = load_file(f"{data_folder}/test.tsv")

    # Merge all into one DataFrame
    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    print(f"✅ Loaded GoEmotions. Total samples: {len(full_df)}")
    
    return full_df
