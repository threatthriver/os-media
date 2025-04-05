"""
Tokenizer for Indian languages supporting major Indian scripts and English.
This tokenizer is designed to handle code-mixed text in Indian languages.
"""

import os
import json
import logging
import sentencepiece as spm
from typing import List, Dict, Any, Optional, Union
import regex as re

logger = logging.getLogger("LLM-Trainer.Tokenizer")

class IndianLanguageTokenizer:
    """
    Tokenizer for Indian languages with support for major Indian scripts.
    Uses SentencePiece under the hood with a vocabulary optimized for Indian languages.
    """
    
    def __init__(
        self,
        vocab_size: int = 100000,  # Larger vocabulary to accommodate Indian scripts
        model_path: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        
        self.special_tokens = [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.mask_token
        ]
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path + ".model"):
            logger.info(f"Loading tokenizer from {model_path}")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_path + ".model")
            
            # Load vocabulary
            self.vocab = {}
            for i in range(self.sp_model.GetPieceSize()):
                piece = self.sp_model.IdToPiece(i)
                self.vocab[piece] = i
                
            logger.info(f"Loaded tokenizer with vocabulary size {len(self.vocab)}")
        else:
            self.sp_model = None
            self.vocab = {}
            logger.warning("No tokenizer model loaded. Train or load a model before using.")
    
    def train(
        self,
        files: List[str],
        output_dir: str,
        model_prefix: str = "indian_tokenizer",
        character_coverage: float = 0.9999,  # High coverage for diverse scripts
        input_sentence_size: int = 10000000,
        shuffle_input_sentence: bool = True,
        normalization_rule_name: str = "nmt_nfkc_cf",  # Standard normalization
        user_defined_symbols: Optional[List[str]] = None,
    ):
        """Train a SentencePiece model on the provided files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join(output_dir, model_prefix)
        
        # Add special tokens to user defined symbols
        if user_defined_symbols is None:
            user_defined_symbols = []
        user_defined_symbols.extend(self.special_tokens)
        
        # Create training command
        train_args = {
            "input": ",".join(files),
            "model_prefix": model_path,
            "vocab_size": self.vocab_size,
            "character_coverage": character_coverage,
            "input_sentence_size": input_sentence_size,
            "shuffle_input_sentence": shuffle_input_sentence,
            "normalization_rule_name": normalization_rule_name,
            "user_defined_symbols": ",".join(user_defined_symbols),
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 3,
            "unk_id": 0,
            "bos_piece": self.bos_token,
            "eos_piece": self.eos_token,
            "pad_piece": self.pad_token,
            "unk_piece": self.unk_token,
            "control_symbols": self.mask_token,
            "byte_fallback": True,  # Fallback to bytes for unknown characters
            "add_dummy_prefix": True,
            "remove_extra_whitespaces": False,  # Important for code
            "train_extremely_large_corpus": True,  # For large datasets
        }
        
        # Convert args to command line format
        train_args_str = " ".join([f"--{k}={v}" for k, v in train_args.items()])
        
        logger.info(f"Training tokenizer with args: {train_args_str}")
        spm.SentencePieceTrainer.Train(train_args_str)
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path + ".model")
        
        # Load vocabulary
        self.vocab = {}
        for i in range(self.sp_model.GetPieceSize()):
            piece = self.sp_model.IdToPiece(i)
            self.vocab[piece] = i
            
        logger.info(f"Trained tokenizer with vocabulary size {len(self.vocab)}")
        
        # Save special token ids
        self.unk_token_id = self.sp_model.PieceToId(self.unk_token)
        self.bos_token_id = self.sp_model.PieceToId(self.bos_token)
        self.eos_token_id = self.sp_model.PieceToId(self.eos_token)
        self.pad_token_id = self.sp_model.PieceToId(self.pad_token)
        self.mask_token_id = self.sp_model.PieceToId(self.mask_token)
        
        return model_path
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token ids."""
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        # Encode text
        ids = self.sp_model.EncodeAsIds(text)
        
        # Add special tokens
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        
        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            if add_eos:
                ids = ids[:max_length - 1] + [self.eos_token_id]
            else:
                ids = ids[:max_length]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = [
                self.sp_model.PieceToId(token) for token in self.special_tokens
                if self.sp_model.PieceToId(token) != self.unk_token_id
            ]
            ids = [id for id in ids if id not in special_ids]
        
        # Decode ids
        text = self.sp_model.DecodeIds(ids)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens."""
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        tokens = self.sp_model.EncodeAsPieces(text)
        return tokens
    
    def save(self, path: str):
        """Save tokenizer model and configuration."""
        if self.sp_model is None:
            raise ValueError("Tokenizer model not loaded. Train or load a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(f"{path}.model", "wb") as f:
            f.write(self.sp_model.serialized_model_proto())
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unk_token_id": self.sp_model.PieceToId(self.unk_token),
            "bos_token_id": self.sp_model.PieceToId(self.bos_token),
            "eos_token_id": self.sp_model.PieceToId(self.eos_token),
            "pad_token_id": self.sp_model.PieceToId(self.pad_token),
            "mask_token_id": self.sp_model.PieceToId(self.mask_token),
        }
        
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved tokenizer to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from pretrained model."""
        # Load configuration
        with open(f"{path}.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            model_path=path,
            unk_token=config["unk_token"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            pad_token=config["pad_token"],
            mask_token=config["mask_token"],
        )
        
        return tokenizer

def create_tokenizer(
    train_files: Optional[List[str]] = None,
    output_dir: str = "./tokenizer",
    model_prefix: str = "indian_tokenizer",
    vocab_size: int = 100000,
    pretrained_path: Optional[str] = None,
):
    """Create or load a tokenizer for Indian languages."""
    if pretrained_path and os.path.exists(pretrained_path + ".model"):
        logger.info(f"Loading pretrained tokenizer from {pretrained_path}")
        return IndianLanguageTokenizer.from_pretrained(pretrained_path)
    
    elif train_files:
        logger.info(f"Training new tokenizer on {len(train_files)} files")
        tokenizer = IndianLanguageTokenizer(vocab_size=vocab_size)
        tokenizer.train(
            files=train_files,
            output_dir=output_dir,
            model_prefix=model_prefix,
        )
        return tokenizer
    
    else:
        raise ValueError("Either train_files or pretrained_path must be provided")
