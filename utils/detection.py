import math
import os
import pdb
import pickle
import random
import re

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support, auc
from sklearn.covariance import LedoitWolf, MinCovDet, GraphicalLasso, OAS

from utils.miscellaneous import save_pkl, load_pkl, return_cov_estimator


# Forward using train data to get empirical mean and covariance
def get_stats(features, labels, cov_estim_name=None, use_shared_cov=False, params=None):
  # Compute mean and covariance of each cls.
  stats = []
  estimators = []
  label_list = range(len(np.unique(labels)))

  if use_shared_cov :
    shared_cov = None
    shared_feat = []

    for idx, lab in enumerate(label_list):
      feat = features[labels==lab]
      shared_feat.append(feat)
      feat = feat
      mu = feat.mean(axis=0)
      stats.append([mu, 0])

    shared_feat = np.concatenate(shared_feat)
    shared_cov = np.cov(shared_feat.T)

    for idx, lab in enumerate(label_list):
      stats[idx][1] = shared_cov

    return stats

  # Estimate covariance per class
  else:
    for idx, lab in enumerate(label_list):
      cov_estim = return_cov_estimator(cov_estim_name, params)
      feat = features[labels==lab]
      mu = feat.mean(axis=0)
      if cov_estim:
        cov = cov_estim.fit(feat).covariance_
        estimators.append(cov_estim)
      else:
        cov = np.cov(feat, rowvar=False)
      stats.append([mu, cov])
  return stats, estimators


def get_train_features(model_wrapper, args, batch_size, dataset, text_key, feat_type='cls', layer=-1):
  assert layer=='pooled' or layer < 0 , "Layer either has to be a int between -12~-1 or the pooling layer"
  model_name = os.path.basename(args.target_model)
  model_name += f"-layer_{layer}"
  if os.path.exists(f"saved_feats/{model_name}_{feat_type}.pkl"):
    features = load_pkl(f"saved_feats/{model_name}_{feat_type}.pkl")
    feats_tensor = []
    for cls, x in enumerate(features):
      data = torch.cat(x, dim=0)
      cls_vector = torch.tensor(cls).repeat(data.shape[0], 1)
      feats_tensor.append(torch.cat([data, cls_vector], dim=1))

    return torch.cat(feats_tensor, dim=0), feats_tensor
  print("Building train features")
  model = model_wrapper.model
  num_samples = len(dataset['label'])
  label_list = np.unique(dataset['label'])
  num_labels = len(label_list)
  num_batches = num_samples // batch_size + 1 if not num_samples % batch_size == 0 else int(num_samples / batch_size)
  features = [[] for _ in range(num_labels)]
  pred_labels = []
  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      if isinstance(text_key, tuple):
        examples = tuple()
        for k in text_key:
          examples += (dataset[k][lower:upper],)
      else:
        examples = dataset[text_key][lower:upper]
      labels = dataset['label'][lower:upper]
      y = torch.LongTensor(labels)
      output, masks = model_wrapper.inference(examples, output_hs=True, output_attention=True)
      prob = F.softmax(output.logits, dim=-1)
      pred = torch.max(prob, dim=-1)[1].detach().cpu().numpy().tolist()
      pred_labels += pred
      preds = output.logits
      if type(layer) == int:
        if feat_type == 'cls':
          feat = output.hidden_states[layer][:, 0, :].cpu()  # (Batch_size, 768)
        elif feat_type == 'mean':
          feat = torch.max(output.hidden_states[layer][:, :, :] * masks, dim=1)[0].cpu()  # (Batch_size, 768)
        for idx, lab in enumerate(label_list):
          features[idx].append(feat[y == lab])
      elif layer == 'pooled':
        feat = output.hidden_states[-1]  # (Batch_size, 768)
        pooled_feat = model.bert.pooler(feat).cpu()
        for idx, lab in enumerate(label_list):

          features[idx].append(pooled_feat[y==lab])
  acc = 0.0
  for p, g in zip(pred_labels, dataset['label']):
    if p == g:
      acc += 1
  print(acc / num_samples)
  if not os.path.exists("saved_feats/"):
    os.mkdir('saved_feats')
  save_pkl(features, f"saved_feats/{model_name}_{feat_type}.pkl")

  feats_tensor = []
  for cls, x in enumerate(features):
    data = torch.cat(x, dim=0)
    cls_vector = torch.tensor(cls).repeat(data.shape[0], 1)
    feats_tensor.append(torch.cat([data, cls_vector], dim=1))
  return torch.cat(feats_tensor, dim=0), feats_tensor


def get_test_features(model_wrapper, batch_size, dataset, text_key, params, logger=None, feat_type='cls', return_probs=False):
  # dataset, batch_size, i, layer = testset['text'].tolist(), 32, 0, -1
  assert logger is not None, "No logger given"
  num_samples = len(dataset[0]) if isinstance(text_key, tuple) else len(dataset)
  num_batches = num_samples // batch_size + 1 if not num_samples % batch_size == 0 else int(num_samples / batch_size)
  features = []
  preds = []
  probs = []
  layer = params['layer_param']['cls_layer']

  with torch.no_grad():
    for i in tqdm(range(num_batches)):
      lower = i * batch_size
      upper = min((i + 1) * batch_size, num_samples)
      if isinstance(text_key, tuple):
        examples = tuple()
        examples += (dataset[0][lower:upper], dataset[1][lower:upper])
      else:
        examples = dataset[lower:upper]
      output, masks = model_wrapper.inference(examples, output_hs=True, output_attention=True)
      prob = F.softmax(output.logits, dim=-1)
      _, pred = torch.max(output.logits, dim=1)
      if feat_type == 'cls':
        feat = output.hidden_states[layer][:, 0, :].cpu()  # (Batch_size, 768)
      elif feat_type == 'mean':
        feat = torch.mean(output.hidden_states[layer][:, :, :] * masks, dim=1).cpu()  # (Batch_size, 768)
      # feat = output.hidden_states[layer][:, 0,:].cpu()  # output.hidden_states : (Batch_size, sequence_length, hidden_dim)
      features.append(feat.cpu())
      preds.append(pred.cpu())
      probs.append(prob.cpu())
  if return_probs:
    return torch.cat(features, dim=0), torch.cat(preds, dim=0), torch.cat(probs, dim=0)
  else:
    return torch.cat(features, dim=0), torch.cat(preds, dim=0)


def detect_attack(testset, confidence, fpr_thres=0.05, visualize=False, logger=None, mode=None,
                  log_metric=False):
  """
  Detect attack for correct samples only to compute detection metric (TPR, recall, precision)
  """
  assert logger is not None, "Logger not given"
  target = np.array(testset['result_type'].tolist())
  conf = confidence.numpy()
  testset['negative_conf'] = -conf # negative of confidence : likelihood of adv. probability
  # Class-agnostic
  fpr, tpr, thres1 = roc_curve(target, -conf)
  precision, recall, thres2 = precision_recall_curve(target, -conf)

  mask = (fpr <= fpr_thres)
  tpr_at_fpr = np.max(tpr * mask) # Maximum tpr at fpr <= fpr_thres
  roc_cutoff = np.sort(np.unique(mask*thres1))[1]
  pred = np.zeros_like(conf)
  pred[-conf>=roc_cutoff] = 1
  prec, rec, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
  auc_value = auc(fpr, tpr)
  logger.log.info(f"TPR at FPR={fpr_thres} : {tpr_at_fpr:.3f}")
  logger.log.info(f"F1 : {f1:.3f}, prec: {prec:.3f}, recall: {rec:.3f}")
  logger.log.info(f"AUC: {auc_value:.3f}")
  if visualize:
    fig, ax = plt.subplots()
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=50, density=False)
    x1 = testset.loc[testset.result_type==0, ['negative_conf']].values.squeeze()
    ax.hist(x=x1, label='clean', **kwargs)
    x2 = testset.loc[testset.result_type==1, ['negative_conf']].values.squeeze()
    ax.hist(x=x2, label='adv', **kwargs)
    ax.annotate(f'{int(roc_cutoff)}', xy=(roc_cutoff,0), xytext=(roc_cutoff,30), fontsize=14,
                arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=3))
    ax.legend()
    fig.savefig(os.path.join(logger.log_path, f"{mode}-hist.png"))

  if log_metric:
    metrics = {"tpr":tpr_at_fpr, "fpr":fpr_thres, "f1":f1, "auc":auc_value}
    logger.log_metric(metrics)

  metric1 = (fpr, tpr, thres1)
  metric2 = (precision, recall, thres2)
  return metric1, metric2, tpr_at_fpr, f1, auc_value


def compute_ppl(texts):
  MODEL_ID = 'gpt2-large'
  print(f"Initializing {MODEL_ID}")
  model = GPT2LMHeadModel.from_pretrained(MODEL_ID).cuda()
  model.eval()
  tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
  encodings = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True)

  batch_size = 1
  num_batch = len(texts) // batch_size
  eval_loss = 0
  likelihoods = []

  with torch.no_grad():
    for i in range(num_batch):
      start_idx = i * batch_size;
      end_idx = (i + 1) * batch_size
      x = encodings[start_idx:end_idx]
      ids = torch.LongTensor(x[0].ids)
      ids = ids.cuda()
      nll = model(input_ids=ids, labels=ids)[0] # negative log-likelihood
      likelihoods.append(-1 * nll.item())

  return torch.tensor(likelihoods)

def augment_data(texts, num_samples, prob_list, ignore_words=['<SPLIT>']):
  ### augment each datapoint with its neighbors
  num_probs = len(prob_list)
  samples_per_prob = num_samples // num_probs
  indices = []
  aug_texts = []
  for idx, text in tqdm(enumerate(texts[:])):
    text = text.split()
    if '<SPLIT>' in text:
      indice = text.index('<SPLIT>')
    else:
      indice = 0
    for prob in prob_list:
      for _ in range(samples_per_prob):
        masked_text = [t if (random.uniform(0, 1) > prob or t in ignore_words or idx <= indice) else '[MASK]' for idx, t in enumerate(text)]
        aug_texts.append(' '.join(masked_text))

    indices.append(len(aug_texts))
  indices.insert(0, 0)
  return aug_texts, indices

def split_text(texts):
  premise = []
  hypothesis = []
  for idx, text in enumerate(texts):
    text_a, text_b = text.split('<SPLIT>')
    premise.append(text_a)
    hypothesis.append(text_b)
  texts = (premise, hypothesis)
  return texts

def pvalue_score(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=5):
    eps = 1e-16
    n_samp = scores_null.shape[0]
    n_obs = scores_obs.shape[0]
    p = np.zeros(n_obs)
    for i in range(n_obs):
        for j in range(n_samp):
            if scores_null[j] >= scores_obs[i]:
                p[i] += 1.

        p[i] = p[i] / n_samp

    if bootstrap:
        ind_null_repl = np.random.choice(np.arange(n_samp), size=(n_bootstrap, n_samp), replace=True)
        p_sum = p
        for b in range(n_bootstrap):
            print(b)
            p_curr = np.zeros(n_obs)
            for i in range(n_obs):
                for j in ind_null_repl[b, :]:
                    if scores_null[j] >= scores_obs[i]:
                        p_curr[i] += 1.

                p_curr[i] = p_curr[i] / n_samp

            p_sum += p_curr

        # Average p-value from the bootstrap replications
        p = p_sum / (n_bootstrap + 1.)

    p[p < eps] = eps
    if log_transform:
        return -np.log(p)
    else:
        return p
