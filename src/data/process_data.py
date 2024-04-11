import pandas as pd
import os

def process_data(directory, train_file, output_directory):
    train_df = pd.read_csv(train_file)
    train_df['segment_id'] = train_df['segment_id'].astype(str)
    time_to_eruption_dict = dict(zip(train_df['segment_id'], train_df['time_to_eruption']))

    dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            media_global = df.mean().mean()
            volcan_id = os.path.splitext(filename)[0]
            volcan_data = {'volcan_id': volcan_id, 'media_global': media_global}
            dfs.append(volcan_data)

    df_global = pd.DataFrame(dfs)
    df_global.set_index('volcan_id', inplace=True)
    df_global['time_to_eruption'] = [time_to_eruption_dict.get(segment_id) for segment_id in df_global.index]

    output_file = 'dataframe.csv'
    index = 1
    while os.path.exists(os.path.join(output_directory, output_file)):
        output_file = f'dataframe{index}.csv'
        index += 1

    output_path = os.path.join(output_directory, output_file)
    df_global.to_csv(output_path, index=True)

directory = 'prueba'
train_file = 'kaggle/input/train.csv'
output_directory = 'processed'
process_data(directory, train_file, output_directory)


