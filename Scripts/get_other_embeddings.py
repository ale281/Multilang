from tqdm import tqdm
import pandas as pd

import os
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

import argparse

def main(model_path, language, force):
    df = pd.read_csv(f"ntcir17_mednlp-sc_sm_train_26_06_23/ntcir17_mednlp-sc_sm_{language}_train_26_06_23.csv")
    # Per  creare il dataset delle label
    dictionary = []
    for c in df.columns[2:]:
        dictionary.append(c.split(":")[1])
    dictionary = pd.DataFrame(dictionary, columns=['AE'])

    dataset = Dataset.from_pandas(dictionary)
    print("aaaaa")

    def tokenize_function(examples):

        return tokenizer(examples["AE"], padding='longest')

    def embed_function(examples):

        del_keys = [k for k in examples if k not in
                    ["input_ids", "token_type_ids", "attention_mask"]]
        for k in del_keys:
            del examples[k]
        examples = {k:v.to(device) for k,v in examples.items()}
        with torch.no_grad():
            out = model(**examples)[0].cpu()
        return dict(CLS=out[:,0,:].numpy(), AVG=out.mean(axis=1).numpy())

    original_model_path = model_path
    
    model_path = "local_"+model_path.replace("/", "|")
    
    output_path = f"{model_path}/{language}_AE_embed_coder.pkl"
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        output_path = f'{model_path}/{language}_AE_embed_coder.pkl'

    if os.path.exists(output_path) and not force:
        print("embeddings already exist at", output_path)
        return

    model = AutoModel.from_pretrained(original_model_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, config=model.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(">> tokenizing")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_dataset.set_format("torch",
                                 columns=["input_ids", "token_type_ids", "attention_mask"] if
                                 "token_type_ids" in tokenized_dataset.format["columns"] else
                                 ["input_ids", "attention_mask"], output_all_columns=True)

    print(">> embedding")
    tokenized_dataset = tokenized_dataset.map(embed_function, batched=True)

    dictionary["coder_cls"] = None
    dictionary["coder_avg"] = None
    print(dictionary.columns)
    print(">> transfering to dataframe")
    for sample, x in zip(tqdm(tokenized_dataset), range(0, len(dictionary))):

        dictionary.iat[x,1] = sample["CLS"]
        dictionary.iat[x,2] = sample["AVG"]
    
    dictionary.to_pickle(output_path)
    print("output serialized to", output_path)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path of the coder model")
    parser.add_argument("-l", "--language", required=True, choices=["en","fr","de","ja"], help="language of the embeddings")
    args = parser.parse_args()
    
    main(args.path, args.language, False)

#python get_other_embeddings.py -p  en_md_1 -l  en