from datasets import load_dataset
import pandas as pd
from utils.logging import get_logger
import os
from transformers import AutoTokenizer

import sys

logger = get_logger(__name__)


class LoadDataset:
    def __init__(self, task, data_path, latest_model) -> None:
        self.task = task
        self.data_path = data_path
        self.latest_model = latest_model
        self.input_feature = None
        self.output_feature = None

    def translation(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "wmt16", "ro-en", split="test", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["input_string"] = test_data_df["translation"].apply(
                    lambda x: x["en"])
                test_data_df["ro"] = test_data_df["translation"].apply(
                    lambda x: x["ro"])
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "ro"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def fill_mask(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "rcds/wikipedia-for-mask-filling", "original_512", split="train", streaming=True
                )
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["title"] = test_data_df["masks"].apply(
                    lambda x: x[0] if len(x) > 0 else ""
                )
                model_name = self.latest_model.name
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                test_data_df["input_string"] = test_data_df["texts"].apply(lambda x: tokenizer.decode(
                    tokenizer.encode(x.replace("<mask>", tokenizer.mask_token), max_length=500, truncation=True,)))
                test_data_fil_df = test_data_df[test_data_df["input_string"].str.contains(
                    tokenizer.mask_token)].reset_index(drop=True)
                with open(self.data_path, "w") as f:
                    f.write(test_data_fil_df.to_json(
                        lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "title"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def question_answering(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "squad_v2", split="validation", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["answer_text"] = test_data_df["answers"].apply(
                    lambda x: x["text"][0] if len(x["text"]) > 0 else "")
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "context,question"
            self.output_feature = "answer_text"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def text_summarization(self):
        try:
            logger.info(f"Data path is this one : {self.data_path} and result is this : {os.path.exists(self.data_path)}")
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "cnn_dailymail", "3.0.0", split="test", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["input_string"] = test_data_df["article"]
                test_data_df["summary"] = test_data_df["highlights"]
                # trucating the data to pass the tokenizer limit of the model
                test_data_df["article"] = test_data_df["article"].str.slice(0, 200)
                test_data_df["input_string"] = test_data_df["input_string"].str.slice(0, 200)
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "summary"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def text_classification(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "glue", "mnli", split="validation_matched", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                id2label = {"id2label": {"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}, "label2id": {
                    "ENTAILMENT": 0, "CONTRADICTION": 2, "NEUTRAL": 1}, }
                id2label = id2label["id2label"]
                label_df = pd.DataFrame.from_dict(
                    id2label, orient="index", columns=["label_string"])
                label_df["label"] = label_df.index.astype("int64")
                label_df = label_df[["label", "label_string"]]
                # join the train, validation and test dataframes with the id2label dataframe to get the label_string column
                test_data_df = test_data_df.merge(label_df, on="label", how="left")
                # concat the premise and hypothesis columns to with "[CLS]" in the beginning and "[SEP]" in the middle and end to get the text column
                test_data_df["input_string"] = "[CLS] " + test_data_df["premise"] + \
                    " [SEP] " + test_data_df["hypothesis"] + " [SEP]"
                # drop the idx, premise and hypothesis columns as they are not needed
                test_data_df = test_data_df.drop(
                    columns=["idx", "premise", "hypothesis"])
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "label_string"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def text_generation(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "cnn_dailymail", "3.0.0", split="test", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                test_data_df["input_string"] = test_data_df["article"].apply(
                    lambda x: x[:100])
                test_data_df["ground_truth"] = test_data_df["article"]
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "ground_truth"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")

    def token_classification(self):
        try:
            if not os.path.exists(self.data_path):
                hf_test_data = load_dataset(
                    "conll2003", split="test", streaming=True)
                test_data_df = pd.DataFrame(hf_test_data.take(100))
                label_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3,
                            "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8, }
                label_reverse_dict = {value: key for key,
                                    value in label_dict.items()}
                test_data_df["input_string"] = test_data_df["tokens"].apply(
                    lambda x: " ".join(x))
                test_data_df["ner_tags_str"] = test_data_df["ner_tags"].apply(
                    lambda x: str([label_reverse_dict[tag] for tag in x]))
                with open(self.data_path, "w") as f:
                    f.write(test_data_df.to_json(lines=True, orient="records"))
                # test_data_df.to_json(self.data_path, lines=True, orient="records")
                df = pd.read_json(self.data_path, lines=True)
                logger.info(f"Here is ths value{df.head(2)}")
            self.input_feature = "input_string"
            self.output_feature = "ner_tags_str"
            return self.input_feature, self.output_feature
        except Exception as ex:
            _, _, exc_tb = sys.exc_info()
            logger.error(
                f"::Error:: Error occuring at this line number : {exc_tb.tb_lineno}")
            logger.error(
                f"::Error:: Error occuring while downloading the datasets and the exception is this : \n {ex}")
