"""
Based on repository of "Mozes, Maximilian, et al. "Frequency-guided word substitutions for detecting textual adversarial examples." EACL (2021)."
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
from utils.preprocess import *

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

class BertWrapper:
    def __init__(self, config, logger):
        self.config = config
        max_seq_len_dict = {"imdb":512, "sst2":64, "ag-news":256, "snli": 64, "paws": 64, 'mnli': 64, 'hate': 64, 'yelp': 128}
        self.max_seq_len = max_seq_len_dict[config.dataset]
        self.seq_len_cache = []
        num_classes = {"imdb":2, "sst2":2, "ag-news":4, "snli": 3, "paws": 2, "yelp": 2, 'mnli': 3,  'hate': 2}
        self.num_classes = num_classes[config.dataset]
        logger.log.info(f"Loading {config.target_model}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.target_model,
            num_labels=num_classes[config.dataset],
            output_attentions=False,
            output_hidden_states=False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.target_model, use_fast=True)

        if len(config.gpu) > 1 :
            device = [torch.device(int(gpu_id)) for gpu_id in config.gpu.split()]
            self.model = torch.nn.DataParallel(self.model, device_ids=device, output_device=device[-1]).cuda()
        elif len(config.gpu) == 1:
            device = torch.device(int(config.gpu))
            self.model = self.model.to(device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.device("cuda")
        else:
            self.device = self.model.device


    def __pre_process(self, text):
        assert isinstance(text, list)
        if isinstance(text[0], list):
            text = [" ".join(t) for t in text]

        if self.config.preprocess == 'fgws':
            text = [fgws_preprocess(t) for t in text]
        elif self.config.preprocess == 'standard':
            pass

        return text

    def inference(self, text, output_hs=False, output_attention=False):
        masks = None
        if isinstance(text, tuple):
            input = tuple()
            for text1 in text:
                input += (self.__pre_process(text1), )
            x = self.tokenizer(*input, max_length=self.max_seq_len, add_special_tokens=True, padding="max_length", truncation=True,
                          return_attention_mask=True, return_tensors='pt')
            masks = self.get_second_sentence_mask(x)
        else:
            text = self.__pre_process(text)
            x = self.tokenizer(text, max_length=self.max_seq_len, add_special_tokens=True, padding="max_length", truncation=True,
                          return_attention_mask=True, return_tensors='pt')
        self.seq_len_cache.extend(x.attention_mask.sum(-1).tolist())
        output = self.model(input_ids=x['input_ids'].to(self.device), attention_mask=x['attention_mask'].to(self.device),
                       token_type_ids=(x['token_type_ids'].to(self.device) if 'token_type_ids' in x else None),
                       output_hidden_states=output_hs, output_attentions=output_attention)

        return output, masks

    def get_second_sentence_mask(self, x):
        return x['token_type_ids'].unsqueeze(-1).to(self.device)

    def get_att_mask(self, text):
        text = self.__pre_process(text)
        x = self.tokenizer(text, max_length=self.max_seq_len, add_special_tokens=True, padding="max_length", truncation=True,
                      return_attention_mask=True, return_tensors='pt')
        return x['attention_mask'].to(self.device)
