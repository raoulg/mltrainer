from __future__ import annotations

import re
import string
from collections import Counter, OrderedDict
from typing import Callable, Optional

import gin
import tokenizers as tk
import torch
from loguru import logger
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab

from mltrainer.preprocessors import PreprocessorProtocol

Tensor = torch.Tensor


def buildBPE(corpus: list[str], vocab_size: int) -> tk.Tokenizer:
    tokenizer = tk.Tokenizer(tk.models.BPE())  # type: ignore
    trainer = tk.trainers.BpeTrainer(  # type: ignore
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<s>", "</s>", "<mask>"],
    )

    # handle spaces better by removing the prefix space
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore
    tokenizer.decoder = tk.decoders.ByteLevel()  # type: ignore

    # train the BPE model
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    return tokenizer


def split_and_flat(corpus: list[str]) -> list[str]:
    corpus_ = [x.split() for x in corpus]
    corpus = [x for y in corpus_ for x in y]
    return corpus


@gin.configurable
def build_vocab(
    corpus: list[str], max: int, oov: str = "<OOV>", pad: str = "<PAD>"
) -> Vocab:
    data = split_and_flat(corpus)
    counter = Counter(data).most_common()
    logger.info(f"Found {len(counter)} tokens")
    counter = counter[: max - 2]
    ordered_dict = OrderedDict(counter)
    v1 = vocab(ordered_dict, specials=[pad, oov])
    v1.set_default_index(v1[oov])
    return v1


def tokenize(corpus: list[str], v: Vocab) -> Tensor:
    batch = []
    for sent in corpus:
        batch.append(torch.tensor([v[word] for word in sent.split()]))
    return pad_sequence(batch, batch_first=True)


def clean(text: str) -> str:
    punctuation = f"[{string.punctuation}]"
    # remove CaPiTaLs
    lowercase = text.lower()
    # change don't and isn't into dont and isnt
    neg = re.sub("\\'", "", lowercase)
    # swap html tags for spaces
    html = re.sub("<br />", " ", neg)
    # swap punctuation for spaces
    stripped = re.sub(punctuation, " ", html)
    # remove extra spaces
    spaces = re.sub("  +", " ", stripped)
    return spaces


@gin.configurable
class Preprocessor:
    def __init__(
        self, max: int, vocab: Vocab, clean: Optional[Callable] = None
    ) -> None:
        self.max = max
        self.vocab = vocab
        self.clean = clean

    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            if clean is not None:
                x = self.clean(x)  # type: ignore
            x = x.split()[: self.max]
            tokens = torch.tensor([self.vocab[word] for word in x], dtype=torch.int32)
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)


class BaseTokenizer(PreprocessorProtocol):
    def __init__(
        self,
        traindataset,
        maxvocab,
        maxtokens,
        clean_fn,
    ) -> None:
        self.maxvocab = maxvocab
        self.maxtokens = maxtokens
        self.clean = clean_fn
        self.vocab = self.build_vocab(self.build_corpus(traindataset))

    @staticmethod
    def split_and_flat(corpus: list[str]) -> list[str]:
        """
        Split a list of strings on spaces into a list of lists of strings
        and then flatten the list of lists into a single list of strings.
        eg ["This is a sentence"] -> ["This", "is", "a", "sentence"]
        """
        corpus_ = [x.split() for x in corpus]
        corpus = [x for y in corpus_ for x in y]
        return corpus

    def build_corpus(self, dataset) -> list[str]:
        corpus = []
        for i in range(len(dataset)):
            x = self.clean(dataset[i][0])
            corpus.append(x)
        return corpus

    def build_vocab(
        self, corpus: list[str], oov: str = "<OOV>", pad: str = "<PAD>"
    ) -> Vocab:
        data = self.split_and_flat(corpus)
        counter = Counter(data).most_common()
        logger.info(f"Found {len(counter)} tokens")
        counter = counter[: self.maxvocab - 2]
        ordered_dict = OrderedDict(counter)
        v1 = vocab(ordered_dict, specials=[pad, oov])
        v1.set_default_index(v1[oov])
        return v1

    def cast_label(self, label: str) -> int:
        raise NotImplementedError

    def __call__(self, batch: list) -> tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            if self.clean is not None:
                x = self.clean(x)  # type: ignore
            x = x.split()[: self.maxtokens]
            tokens = torch.tensor([self.vocab[word] for word in x], dtype=torch.int32)
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)


class TokenizerSettings(BaseModel):
    maxvocab: int
    maxtokens: int
    clean_fn: Callable


class IMDBTokenizer(BaseTokenizer):
    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1

    @classmethod
    def fromSettings(cls, traindataset, settings: TokenizerSettings) -> "IMDBTokenizer":
        return cls(
            traindataset=traindataset,
            maxvocab=settings.maxvocab,
            maxtokens=settings.maxtokens,
            clean_fn=settings.clean_fn,
        )

