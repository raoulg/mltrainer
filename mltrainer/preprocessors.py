from collections import Counter, OrderedDict

from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab


class PreprocessorProtocol(Protocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class BasePreprocessor(PreprocessorProtocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        return torch.stack(X), torch.tensor(y)


class PaddedPreprocessor(PreprocessorProtocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
        return X_, torch.tensor(y)


class BaseTokenizer(PreprocessorProtocol):
    def __init__(
        self, traindataset: TextDataset, settings: TextDatasetSettings
    ) -> None:
        self.maxvocab = settings.maxvocab
        self.maxtokens = settings.maxtokens
        self.clean = settings.clean_fn
        self.vocab = self.build_vocab(self.build_corpus(traindataset))

    @staticmethod
    def split_and_flat(corpus: List[str]) -> List[str]:
        """
        Split a list of strings on spaces into a list of lists of strings
        and then flatten the list of lists into a single list of strings.
        eg ["This is a sentence"] -> ["This", "is", "a", "sentence"]
        """
        corpus_ = [x.split() for x in corpus]
        corpus = [x for y in corpus_ for x in y]
        return corpus

    def build_corpus(self, dataset) -> List[str]:
        corpus = []
        for i in range(len(dataset)):
            x = self.clean(dataset[i][0])
            corpus.append(x)
        return corpus

    def build_vocab(
        self, corpus: List[str], oov: str = "<OOV>", pad: str = "<PAD>"
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

    def __call__(self, batch: List) -> Tuple[Tensor, Tensor]:
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


class IMDBTokenizer(BaseTokenizer):
    def __init__(self, traindataset, settings):
        super().__init__(traindataset, settings)

    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1
