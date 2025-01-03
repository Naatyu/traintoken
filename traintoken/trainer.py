import multiprocessing as mp
from collections import Counter
from itertools import pairwise

import regex
from tqdm import tqdm

from traintoken.logger import get_logger

# o1/o3 (?) regex pattern
BASE_PATTERN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

logger = get_logger("TrainToken")


class TrainToken:
    r"""Trainer for OpenAI TikToken tokenizer.

    Args:
        max_vocab_size (int): maximum desired vocabulary size.
        regex_pattern (str, optional): regex pattern to apply for
                                       pretokenization. Defaults to r"\s+".

    """

    def __init__(
        self,
        max_vocab_size: int,
        regex_pattern: str = BASE_PATTERN,
    ) -> None:
        """Initialize trainer with vocabulary size and regex pattern."""
        self.pre_tokenizer = regex.compile(regex_pattern)

        # Check max vocab size
        if max_vocab_size < 2**8:
            msg = "`max_vocab_size` must be at least 256"
            raise ValueError(msg)
        self.max_vocab_size = max_vocab_size

        # Initialize base vocabulary
        self.vocab = {bytes([i]): i for i in range(256)}

    def _texts_to_token_sequences(
        self,
        texts: list[str],
    ) -> list[list[bytes]]:
        words: list[list[bytes]] = []

        for text in texts:
            # Convert text to byte sequences
            words.extend(
                [
                    [bytes([b]) for b in word.encode("utf-8")]
                    for word in self.pre_tokenizer.findall(text)
                ],
            )

        return words

    def _count_pairs_in_chunk(self, text):
        pair_counts = Counter()

        # for text in texts:
        for pair in pairwise(text):
            pair_counts[pair] += 1

        return pair_counts

    def train(self, texts: list[str], n_proc: int = 1):
        # Pre process texts
        words = self._texts_to_token_sequences(texts)

        # Start the merges
        while len(self.vocab) < self.max_vocab_size:
            pair_counts = Counter()
            with mp.Pool(n_proc) as pool:
                results = pool.imap_unordered(
                    self._count_pairs_in_chunk,
                    tqdm(words),
                )

                for count in results:
                    pair_counts.update(count)

            most_common_pair = max(pair_counts, key=lambda x: pair_counts[x])
            token_bytes = most_common_pair[0] + most_common_pair[1]
            rank = len(self.vocab)
            self.vocab[token_bytes] = rank

            # Now merge that most common pair in all the words. That is, update our training data
            # to reflect our decision to make that pair into a new token.
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == most_common_pair:
                        # We found our pair! Merge it
                        new_word.append(token_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i == len(word) - 1:
                    new_word.append(word[i])
                new_words.append(new_word)
            words = new_words

        return self.vocab
