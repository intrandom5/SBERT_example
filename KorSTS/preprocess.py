from transformers import AutoTokenizer
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

def get_tsv_data(tsv_dir: str):
    '''
    tsv_dir : directory of tsv file.
    return : dictionary of data.
    '''
    tsv = {"genre": [], "filename": [], "year": [], "id": [], "score": [], "sentence1": [], "sentence2": []}

    with open(tsv_dir, "r", encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")

            for i, key in enumerate(tsv.keys()):
                tsv[key].append(line[i].replace("\n", ""))
    
    return tsv

def prepare_data(tsv_dict):
    label = tsv_dict["score"]
    sentence1 = tsv_dict["sentence1"]
    sentence2 = tsv_dict["sentence2"]
    x = [(tokenizer.encode(s1), tokenizer.encode(s2)) for s1, s2 in zip(sentence1, sentence2)]
    assert len(x) == len(label)
    return x, label

if __name__ == "__main__":
    train_tsv = get_tsv_data("sts-train.tsv")
    valid_tsv = get_tsv_data("sts-dev.tsv")
    test_tsv = get_tsv_data("sts-test.tsv")

    train_x, train_y = prepare_data(train_tsv)
    valid_x, valid_y = prepare_data(valid_tsv)
    test_x, test_y = prepare_data(test_tsv)


    np.save("train_x.npy", np.array(train_x, dtype='object'))
    np.save("train_y.npy", np.array(train_y, dtype='object'))

    np.save("valid_x.npy", np.array(valid_x, dtype='object'))
    np.save("valid_y.npy", np.array(valid_y, dtype='object'))

    np.save("test_x.npy", np.array(test_x, dtype='object'))
    np.save("test_y.npy", np.array(test_y, dtype='object'))
