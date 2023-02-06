import random
import statistics
import numpy as np
from tqdm import tqdm
from utils.detection import *
from utils.dataset import *
from utils.miscellaneous import *
from utils.dropout_mc import *

class Detector():
  def __init__(self, model_wrapper, train_stats, loader,
               logger, params, modules, use_val=False, dataset=None, seed=0):

    self.loader = loader
    self.params = params
    self.model_wrapper = model_wrapper
    self.logger = logger
    self.stats = train_stats
    self.seed = seed
    self.data = None
    self.dtype = 'val' if use_val else 'test'
    self.batch_size = 1024 if dataset=="yelp" or dataset == 'ag-news' else 256
    if modules is not None:
      self.scaler = modules[0]
      self.dim_reducer = modules[1]
      self.estimators = modules[2]
      self.estimator_name = modules[3]
    self.num_classes = self.model_wrapper.num_classes

  def get_data(self):
    dataset, _ = self.loader.get_attack_from_csv(batch_size=128, dtype=self.dtype, model_wrapper=None)
    adv_count = dataset.result_type.value_counts()[1]
    total = len(dataset)
    self.logger.log.info(f"Percentage of adv. samples :{adv_count}/{total} = {adv_count/total:3f}")
    return dataset

  def test(self, fpr_thres,  pkl_path=None, model_name='bert', feat_type='cls', text_key='text'):
    testset = self.get_data()

    texts = testset['text'].tolist()
    if '<SPLIT>' in texts[0]:
      texts = split_text(texts)
    model_type = 'roberta' if 'roberta' in model_name else 'bert'
    test_features, preds, probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, params=self.params, feat_type=feat_type, text_key=text_key,
                                             logger=self.logger, return_probs=True)

    max_prob_softmax = []
    for i, pred in enumerate(preds.detach().cpu().numpy().tolist()):
      max_prob_softmax.append(np.sort(probs[i].cpu().numpy())[-1] - np.sort(probs[i].cpu().numpy())[-2])

    # Transform test features if necessary (e.g. PCA, scaling)
    if self.dim_reducer:
      test_features = test_features.numpy()
      if self.scaler:
        test_features = self.scaler.transform(test_features)
      test_features = torch.tensor(self.dim_reducer.transform(test_features))

    metric_header = ["tpr", "fpr", "f1", "auc"]
    for name, stats, estim in zip(["MLE", self.estimator_name], self.stats, self.estimators):
      self.logger.log.info("-----Results-----")
      self.logger.log.info(f"Using {name} estimator")
      if estim:
        all_confidences = []
        test_features = test_features.numpy()
        for per_cls_estim in estim:
          dist = per_cls_estim.mahalanobis(test_features).reshape(-1,1)

          all_confidences.append(dist)
        all_confidences = np.concatenate(all_confidences, axis=1)
        confidence = -torch.tensor(all_confidences[np.arange(preds.numel()), preds])  # Use y_pred to determine which class conditional probability to use

      else:
        confidence, conf_indices, conf_all = compute_dist(test_features, stats, use_marginal=False)
        confidence = conf_all[torch.arange(preds.numel()), preds]   # Use y_pred to determine which class conditional probability to use


      num_nans = sum(confidence == -float("Inf"))
      if num_nans != 0:
        self.logger.log.info(f"Warning : {num_nans} Nans in confidence")
        confidence[confidence == -float("inf")] = -1e6
      roc, pr, tpr_at_fpr, f1, auc = detect_attack(testset, confidence,
                                           fpr_thres,
                                           visualize=True, logger=self.logger, mode=f"{name}-estim", log_metric=True)
      self.logger.save_custom_metric(f"{name}-estim.", [tpr_at_fpr, fpr_thres, f1, auc], metric_header)

    return roc, auc, tpr_at_fpr, confidence, testset

  def get_train_stats(self, texts, model_name, dataset_name, feat_type='cls', text_key='test'):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    if os.path.exists(f"saved_feats/{model_name}_{dataset_name}_train_uncertainty.pkl"):
      aug_uncertainty_mean = load_pkl(f"saved_feats/{model_name}_{dataset_name}_train_uncertainty.pkl")
      aug_confidence_mean = load_pkl(f"saved_feats/{model_name}_{dataset_name}_train_probs.pkl")
      return aug_confidence_mean, aug_uncertainty_mean

    aug_texts, indices = augment_data(texts, 100, [0.1, 0.2, 0.3, 0.4], ignore_words=['<SPLIT>'])
    ue_aug_texts, ue_indices = augment_data(texts, 10, [0.1], ignore_words=['<SPLIT>'])

    ### Load or construct removal neighbors
    if '<SPLIT>' in texts[0]:
      texts = split_text(texts)

    if '<SPLIT>' in aug_texts[0]:
      aug_texts = split_text(aug_texts)

    if '<SPLIT>' in ue_aug_texts[0]:
      ue_aug_texts = split_text(ue_aug_texts)


    self.logger.log.info(f"perform stochastic inference")
    test_features, preds, probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, text_key=text_key,
                                                  params=self.params, feat_type=feat_type,
                                                  logger=self.logger, return_probs=True)
    aug_test_features, aug_preds, aug_probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, text_key=text_key,
                                                     dataset=aug_texts, feat_type=feat_type,
                                                     params=self.params,
                                                     logger=self.logger, return_probs=True)


    aug_preds = []
    for i, pred in enumerate(preds.detach().cpu().numpy().tolist()):
      aug_preds += [pred for _ in range(indices[i + 1] - indices[i])]
    aug_confidence = []
    for i, pred in enumerate(aug_preds):
      aug_confidence.append(aug_probs[i][pred])

    convert_dropouts(self.model_wrapper.model, dropout_type='MC')
    activate_mc_dropout(self.model_wrapper.model, activate=True, random=0.2)
    self.model_wrapper.model.eval()
    self.logger.log.info(f"start_inference")
    eval_results = {}
    eval_results["sampled_probabilities"] = []
    eval_results["sampled_answers"] = []
    eval_results["sampled_aug_probabilities"] = []
    eval_results["sampled_aug_answers"] = []
    for _ in tqdm(range(10)):
        test_features, predsxx, probsxx = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts,
                                                      params=self.params, text_key=text_key,
                                                      logger=self.logger, return_probs=True)
        aug_test_features, aug_preds, aug_probs = get_test_features(self.model_wrapper, batch_size=self.batch_size,
                                                         dataset=ue_aug_texts,
                                                         params=self.params, text_key=text_key,
                                                         logger=self.logger, return_probs=True)
        eval_results["sampled_probabilities"].append(probsxx.tolist())
        eval_results["sampled_answers"].append(predsxx.tolist())
        eval_results["sampled_aug_probabilities"].append(aug_probs.tolist())
        eval_results["sampled_aug_answers"].append(aug_preds.tolist())

    activate_mc_dropout(self.model_wrapper.model, activate=False)

    def probability_variance(sampled_probabilities, mean_probabilities=None):
      e2x = np.mean(np.linalg.norm(sampled_probabilities, axis=-1), axis=0)
      ex2 = np.linalg.norm(np.mean(sampled_probabilities, axis=0), axis=-1)
      return e2x - ex2
    print(eval_results["sampled_aug_probabilities"])


    aug_uncertainty = probability_variance(eval_results["sampled_aug_probabilities"])

    uncertainty = probability_variance(eval_results["sampled_probabilities"])

    aug_confidence_mean = []
    for i in range(0, len(indices) - 1):
      aug_confidence_mean.append(1 - np.mean(aug_confidence[indices[i]: indices[i + 1]]))

    aug_uncertainty_mean = []
    for i in range(0, len(ue_indices) - 1):
      aug_uncertainty_mean.append(np.mean(aug_uncertainty[ue_indices[i]: ue_indices[i + 1]]))

    if not os.path.exists("saved_feats/"):
      os.mkdir('saved_feats')
    save_pkl(aug_confidence_mean, f"saved_feats/{model_name}_{dataset_name}_train_probs.pkl")
    save_pkl(aug_uncertainty_mean, f"saved_feats/{model_name}_{dataset_name}_train_uncertainty.pkl")
    return aug_confidence_mean, aug_uncertainty_mean

  def get_ue(self, fpr_thres, feats, training_probs, training_uncertainty, du_aug=100, mu_aug=10, mu_iters=10, text_key='text', feat_type='cls'):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    testset = self.get_data()

    labels = testset['result_type'].tolist()
    texts = testset['text'].tolist()


    self.logger.log.info(f"perform stochastic inference")

    ### Load or construct neighbors
    aug_texts, indices = augment_data(texts, du_aug, [0.1], ignore_words=['<SPLIT>'])
    ue_aug_texts, ue_indices = augment_data(texts, mu_aug, [0.1], ignore_words=['<SPLIT>'])

    if '<SPLIT>' in texts[0]:
      texts = split_text(texts)

    if '<SPLIT>' in aug_texts[0]:
      aug_texts = split_text(aug_texts)

    if '<SPLIT>' in ue_aug_texts[0]:
      ue_aug_texts = split_text(ue_aug_texts)

    test_features, preds, probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, text_key=text_key,
                                                  params=self.params, feat_type=feat_type,
                                                  logger=self.logger, return_probs=True)


    aug_test_features, aug_preds, aug_probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, text_key=text_key,
                                                     dataset=aug_texts, feat_type=feat_type,
                                                     params=self.params,
                                                     logger=self.logger, return_probs=True)


    max_prob_softmax = []
    for i, pred in enumerate(preds.detach().cpu().numpy().tolist()):
      max_prob_softmax.append(1 - probs[i][pred])

    aug_preds = []
    for i, pred in enumerate(preds.detach().cpu().numpy().tolist()):
      aug_preds += [pred for _ in range(indices[i + 1] - indices[i])]

    aug_confidence = []
    for i, pred in enumerate(aug_preds):
      aug_confidence.append(aug_probs[i][pred])

    aug_confidence_mean = []
    for i in range(0, len(indices) - 1):
      aug_confidence_mean.append(1 - np.mean(aug_confidence[indices[i]: indices[i + 1]]))


    convert_dropouts(self.model_wrapper.model, dropout_type='MC')
    activate_mc_dropout(self.model_wrapper.model, activate=True, random=0.2)
    self.model_wrapper.model.eval()
    self.logger.log.info(f"start_inference")
    eval_results = {}
    eval_results["sampled_probabilities"] = []
    eval_results["sampled_answers"] = []
    eval_results["sampled_aug_probabilities"] = []
    eval_results["sampled_aug_answers"] = []
    for i in tqdm(range(mu_iters)):

        test_features, predsxx, probsxx = get_test_features(self.model_wrapper, batch_size=self.batch_size, dataset=texts, text_key=text_key,
                                                      params=self.params,
                                                      logger=self.logger, return_probs=True)
        aug_test_features, aug_preds, aug_probs = get_test_features(self.model_wrapper, batch_size=self.batch_size, text_key=text_key,
                                                         dataset=ue_aug_texts,
                                                         params=self.params,
                                                         logger=self.logger, return_probs=True)
        eval_results["sampled_probabilities"].append(probsxx.tolist())
        eval_results["sampled_answers"].append(predsxx.tolist())
        eval_results["sampled_aug_probabilities"].append(aug_probs.tolist())
        eval_results["sampled_aug_answers"].append(aug_preds.tolist())

    activate_mc_dropout(self.model_wrapper.model, activate=False)

    def probability_variance(sampled_probabilities, mean_probabilities=None):
      e2x = np.mean(np.linalg.norm(sampled_probabilities, axis=-1), axis=0)
      ex2 = np.linalg.norm(np.mean(sampled_probabilities, axis=0), axis=-1)
      return e2x - ex2


    aug_uncertainty = probability_variance(eval_results["sampled_aug_probabilities"])
    uncertainty = probability_variance(eval_results["sampled_probabilities"])
    aug_uncertainty_mean = []
    for i in range(0, len(ue_indices) - 1):
      aug_uncertainty_mean.append(np.mean(aug_uncertainty[ue_indices[i]: ue_indices[i + 1]]))
    
    aug_confidence = -torch.tensor(aug_confidence_mean, dtype=torch.float).cpu()
    max_prob_softmax = -torch.tensor(max_prob_softmax, dtype=torch.float).cpu()
    metric_header = ["tpr", "fpr", "f1", "auc"]

    self.logger.log.info(f"-----Results for Baseline: du------")
    roc, pr, tpr, f1, auc = detect_attack(testset, max_prob_softmax, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)

    self.logger.log.info(f"-----Results for Baseline: Aug du------")
    roc, pr, tpr, f1, auc = detect_attack(testset, aug_confidence, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)

    aug_uncertainty = -torch.tensor(aug_uncertainty_mean, dtype=torch.float).cpu()
    uncertainty = -torch.tensor(uncertainty, dtype=torch.float).cpu()
    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info(f"-----Results for Baseline: mu------")
    roc, pr, tpr, f1, auc = detect_attack(testset, uncertainty, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)
    self.logger.log.info(f"-----Results for Baseline: Aug mu------")
    roc, pr, tpr, f1, auc = detect_attack(testset, aug_uncertainty, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)

    p_value_probs = pvalue_score(np.array(training_probs), np.array(aug_confidence_mean), log_transform=False, bootstrap=False)

    p_value_uncertainty = pvalue_score(np.array(training_uncertainty), np.array(aug_uncertainty_mean), log_transform=False, bootstrap=False)

    combined_scores = np.log(p_value_probs) + np.log(p_value_uncertainty)
    combined_scores = torch.tensor(combined_scores, dtype=torch.float).cpu()
    p_value_probs = torch.tensor(p_value_probs, dtype=torch.float).cpu()
    p_value_uncertainty = torch.tensor(p_value_uncertainty, dtype=torch.float).cpu()

    self.logger.log.info(f"-----Results for Baseline: p_value_du------")
    roc, pr, tpr, f1, auc = detect_attack(testset, p_value_probs, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)
    self.logger.log.info(f"-----Results for Baseline: p_value_mu------")
    roc, pr, tpr, f1, auc = detect_attack(testset, p_value_uncertainty, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)
    self.logger.log.info(f"-----Results for Baseline: combined------")
    roc, pr, tpr, f1, auc = detect_attack(testset, combined_scores, fpr_thres,
                                          visualize=True, logger=self.logger, mode="Baseline:Uncertainty", log_metric=True)
    self.logger.save_custom_metric("ue", [tpr, fpr_thres, f1, auc], metric_header)

  def test_baseline_PPL(self, fpr_thres, pkl_path=None):
    testset = self.get_data()
    texts = testset['text'].tolist()
    if '<SPLIT>' in texts[0]:
      for idx, text in enumerate(texts):

        text_a, text_b = text.split('<SPLIT>')
        texts[idx] = text_b
    confidence = compute_ppl(texts)
    confidence[torch.isnan(confidence)] = 1e6
    confidence[confidence == -float("inf")] = -1e6
    metric_header = ["tpr", "fpr", "f1", "auc"]
    self.logger.log.info("-----Results for Baseline: GPT-2 PPL------")
    roc, pr, tpr, f1, auc = detect_attack(testset, confidence, fpr_thres,
                                visualize=False, logger=self.logger, mode="Baseline:PPL", log_metric=True)
    self.logger.save_custom_metric("ppl", [tpr, fpr_thres, f1, auc], metric_header)

