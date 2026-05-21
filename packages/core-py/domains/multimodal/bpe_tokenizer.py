"""
BPE (Byte-Pair Encoding) Tokenizer for multimodal caption vocabulary.

Trains on caption corpus to build subword vocabulary, enabling the decoder
to handle OOV words and produce more natural captions.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class BPETokenizer:
    """Subword tokenizer using Byte-Pair Encoding.
    
    Trains on caption text to learn merge operations, then encodes
    new text into subword token IDs.
    """
    
    SAVE_PATH = "data/multimodal/bpe_tokenizer.json"
    
    def __init__(self, vocab_size: int = 4096, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]
        self.vocab: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._built = False
    
    def _preprocess(self, text: str) -> List[str]:
        """Split text into initial character/word tokens."""
        text = text.lower().strip()
        # Add word boundary markers
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        # Split each word into characters with end-of-word marker
        tokens = []
        for w in words:
            tokens.extend(list(w[:-1]))
            tokens.append(w[-1] + '</w>')
        return tokens
    
    def _get_stats(self, vocab: Counter) -> Dict[Tuple[str, str], int]:
        """Count adjacent token pairs."""
        pairs = Counter()
        for word, count in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += count
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Counter) -> Counter:
        """Merge the most frequent pair in vocabulary."""
        new_vocab = Counter()
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, count in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = count
        return new_vocab
    
    def train(self, texts: List[str]):
        """Train BPE tokenizer on caption corpus.
        
        Args:
            texts: List of caption strings to learn merges from.
        """
        # Initialize with character-level tokens
        word_vocab = Counter()
        for text in texts:
            tokens = self._preprocess(text)
            word_str = ' '.join(tokens)
            word_vocab[word_str] += 1
        
        # Build initial vocab (unique characters)
        char_vocab = set()
        for word in word_vocab:
            char_vocab.update(word.split())
        
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        for char in sorted(char_vocab):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Learn merges until vocab_size reached
        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pairs = self._get_stats(word_vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break  # Stop if no pair appears more than once
            
            self.merges.append(best_pair)
            merged = ''.join(best_pair)
            self.vocab[merged] = len(self.vocab)
            word_vocab = self._merge_vocab(best_pair, word_vocab)
        
        # Build itos
        self.itos = {i: tok for tok, i in self.vocab.items()}
        self._built = True
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using learned merges."""
        if not self._built:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        tokens = self._preprocess(text)
        token_str = ' '.join(tokens)
        
        # Apply merges
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            token_str = token_str.replace(bigram, replacement)
        
        # Convert to IDs
        ids = []
        for tok in token_str.split():
            ids.append(self.vocab.get(tok, self.vocab.get("<UNK>", 3)))
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            tok = self.itos.get(tid, "")
            if tok in ("<BOS>", "<EOS>", "<PAD>", "<UNK>", ""):
                continue
            tokens.append(tok)
        # Join and clean up word boundary markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ').strip()
        return text
    
    def save(self, path: Optional[str] = None):
        """Save tokenizer state to JSON."""
        path = path or self.SAVE_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Optional[str] = None):
        """Load tokenizer state from JSON."""
        path = path or self.SAVE_PATH
        if not Path(path).exists():
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.itos = {i: tok for tok, i in self.vocab.items()}
        self._built = True
        return True
