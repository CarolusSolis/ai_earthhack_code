import numpy as np
import pandas as pd
from pathlib import Path


ASSISTANT_TOKEN = '<|assistant|>'
if __name__ == '__main__':
    dataset = 'AI EarthHack Dataset.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    verbose_outputs = []
    fails = []

    for idx in range(len(df)):
        problem_id = df.loc[idx, 'id']
        problem = df.loc[idx, 'problem'].strip().replace('\n', '')
        solution = df.loc[idx, 'solution']

        output_file = Path(f'{problem_id}.txt')

        if not problem or not isinstance(problem, str):
            verbose_output = 'Empty problem'
            fail = True
        if not solution or not isinstance(solution, str):
            verbose_output = 'Empty solution'
            fail = True
        elif not output_file.is_file():
            verbose_output = ''
            fail = 'N/A'
        else:
            with open(output_file, 'r') as f:
                output = f.read()

            start_idx = output.find(ASSISTANT_TOKEN) + len(ASSISTANT_TOKEN)
            verbose_output = output[start_idx:].strip()
            fail = 'FAIL' in verbose_output

        verbose_outputs.append(verbose_output)
        fails.append(fail)

    df.insert(len(df.columns), 'Verbose', verbose_outputs)
    df.insert(len(df.columns), 'Failed', fails)
    df.to_csv('filtered_dataset.csv', encoding='latin-1', index=False)
