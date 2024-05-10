from pathlib import Path
import os
import json

base_path = ""
explanations_path = Path(base_path + "data/explanations/")
pcaps_path = Path(base_path + "data/pcaps/")

prompt = """Question: Explain the table, covering login attempts, brute force attacks, certificate issues,
and other network activities. Include causes, detection methods, and mitigation strategies for each scenario. 
Be clear and specific."""

def get_dataset(path:Path):
    dataframe = []
    # go through data sources
    for f in os.listdir(path):
        f = Path(f)
        if f.suffix == ".json":
            data_source_name = Path(f.stem)
            # for each example from that source
            for file in os.listdir(explanations_path / data_source_name):
                example_name = Path(file)
                # add example to dataset
                with open(explanations_path / data_source_name / example_name, encoding="utf8") as fpe:
                    with open(pcaps_path / data_source_name / example_name, encoding="utf8") as fpp:
                            input = f"{prompt}\n{fpp.read()}"
                            output = fpe.read()
                            dataframe.append({"input": input, "output": output})
    # save data
    with open(path / Path("data.json"), "w") as fp:
        json.dump(dataframe, fp)

def prepare_prompt(path: Path):
    with open(path) as fp:
        pcap = fp.read()
        # return prompt with pcap content concatenation
        return f"{prompt}\n{pcap}"
    