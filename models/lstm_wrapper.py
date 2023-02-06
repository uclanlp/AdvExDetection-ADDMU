"""
LSTM 4 Classification
---------------------------------------------------------------------

"""
import json
import os

import torch
from torch import nn as nn

import textattack
from textattack.models.helpers import GloveEmbeddingLayer
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.shared import utils
from textattack.models.wrappers.model_wrapper import ModelWrapper

class LSTMOutput:
    logits = None
    hidden_states = None


class LSTMForClassification(nn.Module):
    """A long short-term memory neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        depth=1,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super().__init__()
        self._config = {
            "architectures": "LSTMForClassification",
            "hidden_size": hidden_size,
            "depth": depth,
            "dropout": dropout,
            "num_labels": num_labels,
            "max_seq_length": max_seq_length,
            "model_path": model_path,
            "emb_layer_trainable": emb_layer_trainable,
        }
        if depth <= 1:
            # Fix error where we ask for non-zero dropout with only 1 layer.
            # nn.module.RNN won't add dropout for the last recurrent layer,
            # so if that's all we have, this will display a warning.
            dropout = 0
        self.drop = nn.Dropout(dropout)
        self.emb_layer_trainable = emb_layer_trainable
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = nn.LSTM(
            input_size=self.emb_layer.n_d,
            hidden_size=hidden_size // 2,
            num_layers=depth,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = textattack.models.tokenizers.GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        self.device = 1
        self.eval()

    def load_from_disk(self, model_path):
        # TODO: Consider removing this in the future as well as loading via `model_path` in `__init__`.
        import warnings

        warnings.warn(
            "`load_from_disk` method is deprecated. Please save and load using `save_pretrained` and `from_pretrained` methods.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_state_dict(load_cached_state_dict(model_path))
        self.eval()

    def save_pretrained(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(
            state_dict,
            os.path.join(output_path, "pytorch_model.bin"),
        )
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self._config, f)

    @classmethod
    def from_pretrained(cls, name_or_path):
        """Load trained LSTM model by name or from path.

        Args:
            name_or_path (:obj:`str`): Name of the model (e.g. "lstm-imdb") or model saved via :meth:`save_pretrained`.
        Returns:
            :class:`~textattack.models.helpers.LSTMForClassification` model
        """
        TEXTATTACK_MODELS = {
            #
            # LSTMs
            #
            "textattack/lstm-ag-news": "models_v2/classification/lstm/ag-news",
            "textattack/lstm-imdb": "models_v2/classification/lstm/imdb",
            "textattack/lstm-mr": "models_v2/classification/lstm/mr",
            "textattack/lstm-sst2": "models_v2/classification/lstm/sst2",
            "textattack/lstm-yelp": "models_v2/classification/lstm/yelp",
            #
            # CNNs
            #
            "cnn-ag-news": "models_v2/classification/cnn/ag-news",
            "cnn-imdb": "models_v2/classification/cnn/imdb",
            "cnn-mr": "models_v2/classification/cnn/rotten-tomatoes",
            "cnn-sst2": "models_v2/classification/cnn/sst",
            "cnn-yelp": "models_v2/classification/cnn/yelp",
            #
            # T5 for translation
            #
            "t5-en-de": "english_to_german",
            "t5-en-fr": "english_to_french",
            "t5-en-ro": "english_to_romanian",
            #
            # T5 for summarization
            #
            "t5-summarization": "summarization",
        }

        if name_or_path in TEXTATTACK_MODELS:
            # path = utils.download_if_needed(TEXTATTACK_MODELS[name_or_path])
            path = utils.download_if_needed(TEXTATTACK_MODELS[name_or_path])
        else:
            path = name_or_path

        config_path = os.path.join(path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "architectures": "LSTMForClassification",
                "hidden_size": 150,
                "depth": 1,
                "dropout": 0.3,
                "num_labels": 2,
                "max_seq_length": 128,
                "model_path": None,
                "emb_layer_trainable": True,
            }
        del config["architectures"]
        model = cls(**config)
        state_dict = load_cached_state_dict(path)
        model.load_state_dict(state_dict)
        return model

    def forward(self, _input, output_hidden=True):
        # ensure RNN module weights are part of single contiguous chunk of memory
        self.encoder.flatten_parameters()

        emb = self.emb_layer(_input)
        emb = self.drop(emb)

        output, hidden = self.encoder(emb)
        output_pooled = torch.max(output, dim=1)[0]

        output_pooled = self.drop(output_pooled)
        pred = self.out(output_pooled)
        return pred, output

    def get_input_embeddings(self):
        return self.emb_layer.embedding

class LSTMWrapper():
    def __init__(self, config, logger):
        self.config = config
        max_seq_len_dict = {"imdb":512, "sst2":64, "ag-news":256, "snli": 64, "paws": 64, 'mnli': 64, 'hate': 64, 'yelp': 128}
        self.max_seq_len = max_seq_len_dict[config.dataset]
        self.seq_len_cache = []
        num_classes = {"imdb":2, "sst2":2, "ag-news":4, "snli": 3, "paws": 2, "yelp": 2, 'mnli': 3,  'hate': 2}
        self.num_classes = num_classes[config.dataset]
        logger.log.info(f"Loading {config.target_model}")

        self.model = LSTMForClassification.from_pretrained(
            config.target_model)
        self.tokenizer = self.model.tokenizer

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

    def encode(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input.
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[int]]): List of list of ids
        """
        if hasattr(self.tokenizer, "batch_encode"):
            return self.tokenizer.batch_encode(inputs)
        else:
            return [self.tokenizer.encode(x) for x in inputs]

    def inference(self, text, output_hs=False, output_attention=False):
        model_device = next(self.model.parameters()).device
        ids = self.encode(text)
        ids = torch.tensor(ids).to(model_device)
        with torch.no_grad():
            outputs, hiddens = self.model(ids)
        output = LSTMOutput()
        output.logits = outputs
        output.hidden_states = [hiddens]
        return output, torch.ones(hiddens.shape).to(model_device)

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
