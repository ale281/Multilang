from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None
import glob
import torch
import os

import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

import argparse

from sklearn.metrics import jaccard_score

from get_other_embeddings import main as get_other_embeddings


def main(model_path, outpath, dataset, split, threshold):
    outpath_avg = f"{outpath}/{dataset}/run_avg"
    outpath_cls = f"{outpath}/{dataset}/run_cls"
    jacc_path = lambda outpath: f"{outpath}/jacc.txt"
    #pred_path = lambda outpath: f"{outpath}/preds.csv"
    pred_path = lambda outpath: f"{outpath}/preds.csv"

    if not os.path.exists(outpath_avg):
        os.makedirs(outpath_avg)
    if not os.path.exists(outpath_cls):
        os.makedirs(outpath_cls)

    if os.path.exists(pred_path(outpath_avg)) and os.path.exists(pred_path(outpath_cls)) and (not args.force):
        print(pred_path(outpath_avg), "and", pred_path(outpath_cls), "already exist")
        return

    #TOKENIZATION AND EMBEDDING
    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COL], padding='longest')

    def embed_function(examples):
        del_keys = [k for k in examples if k not in
                    ["input_ids", "token_type_ids", "attention_mask"]]
        for k in del_keys:
            del examples[k]
        examples = {k:v.to(device) for k,v in examples.items()}
        with torch.no_grad():
            out = model(**examples)[0].cpu()
        return dict(CLS=out[:,0,:].numpy(), AVG=out.mean(axis=1).numpy())

    test_set = f"prep_data/{dataset}.csv"
    
    original_model_path = model_path
    
    model_path = "local_"+model_path.replace("/", "|")

    if not os.path.exists(f"{model_path}/en_AE_embed_coder.pkl") or args.force:
        print("embedding vocabulary first")
        get_other_embeddings(original_model_path, args.force)


    #PRENDI MODELLO E TOKENIZZATORE E PASSA IN CUDA
    model = AutoModel.from_pretrained(original_model_path)
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, config=model.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    #PRENDI GLI EMBEDDING
    AEs = pd.read_pickle(f"{model_path}/en_AE_embed_coder.pkl")

    #PRENDI E PREPARA IL DATASET
    TEXT_COL = "text"
    df1 = pd.read_csv(test_set, index_col=False)
    df1.rename(columns={'train_id': 'id'}, inplace=True)
    cols = df1.columns[4:]
    #print(cols)
    df = df1[["id", "text", "AE_labels"]]

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(float(args.split))
    dataset = dataset['test']

    #TOKENIZZA ED EMBEDDA
    print(">> tokenizing")
    dataset = dataset.map(tokenize_function, batched=True, batch_size=None)
    dataset.set_format(
        "torch",
        columns=["input_ids", "token_type_ids", "attention_mask"] if "token_type_ids" in dataset.format["columns"] else ["input_ids", "attention_mask"],
        output_all_columns=True
    )
    print(">> embedding")
    dataset = dataset.map(embed_function, batched=True, batch_size=256)

    #FUNZIONE DI COS SIMILARITY
    sim_function = torch.nn.CosineSimilarity(dim=1)

    all_AEs_cls = torch.stack(AEs.coder_cls.values.tolist())
    all_AEs_avg = torch.stack(AEs.coder_avg.values.tolist())

    df_avg = df[df["id"].isin(dataset["id"])]#df.copy()
    df_avg["model_generated"] = None

    df_cls = df[df["id"].isin(dataset["id"])]#df.copy()#
    df_cls["model_generated"] = None

    df_true = df1[df1["id"].isin(dataset["id"])]


    for col in cols:
        df_cls[col] = 0
        df_avg[col] = 0
    print(">> dfs ready")

    #Calculates similarity according to a threshold
    for idx,sample in enumerate(tqdm(dataset, desc="calculating similarity")):

        sim = sim_function(sample["CLS"], all_AEs_cls)
        sort_sim, indices = sim.sort(descending=True)
        #print(sort_sim)
        sort_sim = [i for i in sort_sim if i >= args.threshold]
        indices = indices[:len(sort_sim)]
        #indice della colonna :3 = model_generated
        df_cls.iat[idx,3] = [AEs.iloc[i].AE for i in indices.tolist()]

        sim = sim_function(sample["AVG"], all_AEs_avg)
        sort_sim, indices = sim.sort(descending=True)
        #print(sort_sim)
        sort_sim = [i for i in sort_sim if i >= args.threshold]
        indices = indices[:len(sort_sim)]
        # indice della colonna :3 = model_generated
        df_avg.iat[idx, 3] = [AEs.iloc[i].AE for i in indices.tolist()]

    #AGGIUNGI LE LABEL PREDETTE IN FORMA BINARIA
    for x in range(0, len(df_cls)):
        sample_cls = df_cls.iloc[x]
        sample_avg = df_avg.iloc[x]
        aes_cls = sample_cls.model_generated
        aes_avg = sample_avg.model_generated
        for col in cols:
            if aes_cls is not None:
                if col.split(":")[1] in aes_cls:
                    df_cls.at[sample_cls.name, col] = 1

            if aes_avg is not None:
                if col.split(":")[1] in aes_avg:
                    df_avg.at[sample_avg.name, col] = 1

    for gt, data, op in zip([df_true], [df_cls], [outpath_cls]):
        data.to_csv(pred_path(op))
        x = jaccard_score(gt[cols], data[cols], average='samples')
        print("CLS jaccard score is:")
        print(x)
        #x = pd.DataFrame(list(x.values()), index=x.keys())
        #x.to_csv(jacc_path(op), index=False)

    for gt, data, op in zip([df_true], [df_avg], [outpath_avg]):
        data.to_csv(pred_path(op))
        x = jaccard_score(gt[cols], data[cols], average='samples')
        print("AVG jaccard score is:")
        print(x)
        #x = pd.DataFrame(list(x.values()), index=x.keys())
        #x.to_csv(jacc_path(op), index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path of the coder model")
    parser.add_argument("-o", "--outpath", required=True, help="name of the main output dir for this model")
    parser.add_argument("-d", "--dataset", required=True, choices=["en_data", "fr_data", "de_data", "ja_data"], help="dataset to test on")
    parser.add_argument("-s", "--split", required=True, help="data split to test on")
    parser.add_argument("-f", "--force", action="store_true", help="force to recompute embeddings")
    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold to pick similarities")
    args = parser.parse_args()
    
    main(args.path, args.outpath, args.dataset, args.split, args.threshold)
    
# python get_other_predictions.py -p en_md_1 -o results -d en_data -s 0.2 -t 0.5