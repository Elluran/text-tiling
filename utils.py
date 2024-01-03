from functools import cache
from transformers import AutoTokenizer, BertForNextSentencePrediction
from sentence_transformers import SentenceTransformer
from typing import Callable
from torch.nn import functional as F
import pandas as pd
import torch
import numpy as np
from nltk.metrics import windowdiff
from tqdm import tqdm


def depth_smoothing(scores: list[float]) -> list[float]:
    depth_scores = []

    for i in range(len(scores)):
        left_max, right_max = scores[i], scores[i]

        for l in range(i - 1, -1, -1):
            if scores[l] >= left_max:
                left_max = scores[l]
            else:
                break

        for r in range(i + 1, len(scores)):
            if scores[r] >= right_max:
                right_max = scores[r]
            else:
                break

        depth_scores.append((left_max + right_max - 2 * scores[i]) / 2)

    return depth_scores


@cache
def load_bert_model():
    device = torch.device("cuda:0")
    return (
        BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        .to(device)
        .eval()
    )


def scorer_bert(messages: list[str]) -> list[float]:
    scores = []

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = load_bert_model()

    for i in range(len(messages) - 1):
        j = i + 1

        tokenized_input = {}
        if tokenizer:
            tokenized_input = tokenizer(
                messages[i], messages[j], return_tensors="pt"
            ).to(device)

        with torch.no_grad():
            logits = model.forward(**tokenized_input).logits

        score = F.softmax(logits, dim=1)[0][0]
        scores.append(score.item())

    return scores


def scorer_all_mpnet(messages: list[str]) -> list[float]:
    scores = []

    device = torch.device("cuda:0")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
    with torch.no_grad():
        embeddings = model.encode(messages)

    # print(embeddings.shape)
    for i in range(len(messages) - 1):
        cosine_sim = F.cosine_similarity(
            torch.tensor(embeddings[i]), torch.tensor(embeddings[i + 1]), dim=0
        ).item()
        scores.append(cosine_sim)

    return scores


def segment_dialog(
    messages: list[str],
    thr: float = 0.5,
    scorer: Callable[[list[str]], list[float]] = scorer_bert,
    smoothing_strategy: Callable[[list[float]], list[float]] = None,
):
    seg_boundaries = []
    seg_boundaries_mask = []

    scores = scorer(messages)

    if smoothing_strategy is None:
        smooth_scores = np.ones(len(scores)) - np.array(scores)
    elif smoothing_strategy == "depth":
        smooth_scores = depth_smoothing(scores)

    for i in range(len(smooth_scores)):
        if smooth_scores[i] > thr:
            seg_boundaries.append(i + 1)
            seg_boundaries_mask.append(1)
        else:
            seg_boundaries_mask.append(0)

    return np.array(seg_boundaries), np.array(seg_boundaries_mask)


def calculate_mean_windowdiff(
    df: pd.DataFrame,
    scorer: Callable[[list[str]], list[float]] = scorer_bert,
    smoothing_strategy: Callable[[list[float]], list[float]] = None,
    thr: float = 0.5,
):
    acc_sum = 0
    for _, (messages, _, orig_bouds) in tqdm(df.iterrows()):
        _, predicted_bounds = segment_dialog(
            messages, scorer=scorer, smoothing_strategy=smoothing_strategy, thr=thr
        )
        acc_sum += windowdiff(list(predicted_bounds), list(orig_bouds), 3, 1)

    return acc_sum / len(df)


def calculate_random_mean_windowdiff(df):
    acc_sum = 0
    for _, (messages, _, orig_bouds) in df.iterrows():
        _, predicted_bounds = random_segment_dialog(messages)
        # print(predicted_bounds)
        # print(orig_bouds)
        acc_sum += windowdiff(list(predicted_bounds), list(orig_bouds), 3, 1)

    return acc_sum / len(df)


def random_segment_dialog(messages, thr=0.5):
    seg_boundaries = []
    seg_boundaries_mask = []
    torch.manual_seed(42)
    for i in range(len(messages) - 1):
        if torch.rand((1,)) < thr:
            seg_boundaries.append(i + 1)
            seg_boundaries_mask.append(1)
        else:
            seg_boundaries_mask.append(0)
    seg_boundaries.append(len(messages))

    return np.array(seg_boundaries), np.array(seg_boundaries_mask)
