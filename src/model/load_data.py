import pandas as pd
from pathlib import Path
import os

base_path = ""
explanations_path = Path(base_path + "data/explanations/")
pcaps_path = Path(base_path + "data/pcaps/")

prompt = """Question: Explain the table, covering login attempts, brute force attacks, certificate issues,
and other network activities. Include causes, detection methods, and mitigation strategies for each scenario. 
Be clear and specific."""

def get_dataset(path:Path):
    dataframe = []

    for f in os.listdir(path):
        f = Path(f)
        if f.suffix == ".json":
            data_source_name = Path(f.stem)
            for file in os.listdir(explanations_path / data_source_name):
                example_name = Path(file)
                with open(explanations_path / data_source_name / example_name, encoding="utf8") as fpe:
                    with open(pcaps_path / data_source_name / example_name, encoding="utf8") as fpp:
                            example = f"{prompt}\n{fpp.read()}\nAnswer: {fpe.read()}"
                            dataframe.append(example)

    return dataframe