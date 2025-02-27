import os

import pandas as pd


def merge_csv_files(input_folder, output_file):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        filename = os.path.splitext(file)[0]
        df = df.rename(columns={col: f"{filename}_{col}" for col in df.columns if col != "Gold_Labels"})
        dataframes.append(df)

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on='Gold_Labels', how='outer')
    cos_sim_columns = [col for col in merged_df.columns if 'cos_sim_scores' in col]
    merged_df['average_cos_sim_scores'] = merged_df[cos_sim_columns].mean(axis=1, skipna=True)

    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved as {output_file}")


input_folder = "./"
output_file = "merged_output.csv"
merge_csv_files(input_folder, output_file)
