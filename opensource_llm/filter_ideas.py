import sys
import numpy as np
import pandas as pd
from pathlib import Path


ASSISTANT_TOKEN = '<|assistant|>'
EVALUATION_TOKEN = 'EVALUATION:'


def find_failure(output):
    output = output.lower()
    lines = output.split('\n')
    criteria = [
        'overall',
        'sloppy',
        'sloppiness',
        'relevance',
        'off-topic',
        'suitable',
        'suitability',
        'vague',
        'practicality',
        'feasiblity',
    ]
    return np.any([
        ('fail' in i)
            and (np.any([criterion in i for criterion in criteria]))
        for i in lines
    ])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
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
            output = output.strip()

            start_idx = (
                output.find(ASSISTANT_TOKEN) + 1
                + len(ASSISTANT_TOKEN)
                + len(EVALUATION_TOKEN)
                + 1 # newline after assistant token
            )

            output = output[start_idx:].strip()
            end_idx = output.find(EVALUATION_TOKEN)
            if end_idx == -1:
                end_idx = len(output)

            verbose_output = output[:end_idx].strip()

            # fail = 'FAIL' in verbose_output
            fail = find_failure(verbose_output)

        verbose_outputs.append(verbose_output)
        fails.append(fail)

    df.insert(len(df.columns), 'Verbose', verbose_outputs)
    df.insert(len(df.columns), 'Failed', fails)
    df.to_csv('filtered_dataset.csv', encoding='latin-1', index=False)
