"""
Tokenizer for LLM model.
"""

import os
import json
import regex as re
from typing import Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """
    
    @abstractmethod
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        pass
    
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """
        Get beginning of sequence token ID.
        
        Returns:
            Beginning of sequence token ID
        """
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """
        Get end of sequence token ID.
        
        Returns:
            End of sequence token ID
        """
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """
        Get padding token ID.
        
        Returns:
            Padding token ID
        """
        pass


class SentencePieceTokenizer(Tokenizer):
    """
    SentencePiece tokenizer.
    
    Attributes:
        model_path: Path to SentencePiece model
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        pad_id: Padding token ID
    """
    
    def __init__(
        self,
        model_path: str,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to SentencePiece model
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            pad_id: Padding token ID
        """
        self.model_path = model_path
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._pad_id = pad_id
        
        # Load SentencePiece model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token
            
        Returns:
            List of token IDs
        """
        # Encode text
        token_ids = self.sp_model.EncodeAsIds(text)
        
        # Add special tokens
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [
                token_id for token_id in token_ids
                if token_id not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]
            ]
        
        # Decode token IDs
        text = self.sp_model.DecodeIds(token_ids)
        
        return text
    
    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self.sp_model.GetPieceSize()
    
    @property
    def bos_token_id(self) -> int:
        """
        Get beginning of sequence token ID.
        
        Returns:
            Beginning of sequence token ID
        """
        return self._bos_id
    
    @property
    def eos_token_id(self) -> int:
        """
        Get end of sequence token ID.
        
        Returns:
            End of sequence token ID
        """
        return self._eos_id
    
    @property
    def pad_token_id(self) -> int:
        """
        Get padding token ID.
        
        Returns:
            Padding token ID
        """
        return self._pad_id


def train_sentencepiece_tokenizer(
    texts: List[str],
    vocab_size: int,
    model_prefix: str,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    user_defined_symbols: Optional[List[str]] = None
) -> str:
    """
    Train SentencePiece tokenizer.
    
    Args:
        texts: List of training texts
        vocab_size: Vocabulary size
        model_prefix: Prefix for model files
        character_coverage: Character coverage
        model_type: Model type (bpe, unigram, char, word)
        user_defined_symbols: List of user-defined symbols
        
    Returns:
        Path to trained model
    """
    # Write training data to temporary file
    with open(f"{model_prefix}.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Set training arguments
    args = [
        f"--input={model_prefix}.txt",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        f"--model_type={model_type}",
        "--pad_id=0",
        "--bos_id=1",
        "--eos_id=2",
        "--unk_id=3",
        "--pad_piece=<pad>",
        "--bos_piece=<bos>",
        "--eos_piece=<eos>",
        "--unk_piece=<unk>",
        "--normalization_rule_name=nmt_nfkc_cf",
    ]
    
    # Add user-defined symbols if provided
    if user_defined_symbols:
        args.append(f"--user_defined_symbols={','.join(user_defined_symbols)}")
    
    # Train tokenizer
    spm.SentencePieceTrainer.Train(" ".join(args))
    
    # Remove temporary file
    os.remove(f"{model_prefix}.txt")
    
    return f"{model_prefix}.model"
