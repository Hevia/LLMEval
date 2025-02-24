import pandas as pd
import json
import os
from difflib import ndiff

with open("corpus/test.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

dialogue_map = {item["summary"]: item["dialogue"] for item in json_data}

input_dir = "output"
output_dir = "output/with_input_diff"
os.makedirs(output_dir, exist_ok=True)

def get_text_diff(text1, text2):
    return '\n'.join(list(ndiff(str(text1).splitlines(), str(text2).splitlines())))

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_csv_path = os.path.join(input_dir, filename)
        output_csv_path = os.path.join(output_dir, filename)

        df = pd.read_csv(input_csv_path)
        df["Input"] = df["Gold_Labels"].map(dialogue_map)
        df["Text_Diff"] = df.apply(lambda row: get_text_diff(row["Input"], row["Model_Responses"]), axis=1)
        df.to_csv(output_csv_path, index=False)
        print(f"Processed and saved: {output_csv_path}")

print("All CSV files processed successfully.")