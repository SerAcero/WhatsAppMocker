"""
Train an LLM to mimic WhatsApp conversation styles.
Uses LoRA fine-tuning for efficient training on consumer hardware.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from config import Config
from dataset import WhatsAppDataset

os.environ["HF_HOME"] = "D:\\WhatsappMocker\\HuggingFaceCache"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training."""

    base_model: str = Config.BASE_MODEL
    output_dir: str = Config.MODEL_OUTPUT_DIR
    messages_jsonl: str = os.path.join(Config.PROCESSED_DIR, "messages.jsonl")
    personas_json: str = os.path.join(Config.PROCESSED_DIR, "personas.json")

    # Training hyperparameters
    num_epochs: int = Config.NUM_EPOCHS
    batch_size: int = Config.BATCH_SIZE
    gradient_accumulation_steps: int = Config.GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = Config.LEARNING_RATE
    max_seq_length: int = Config.MAX_SEQ_LENGTH
    window_size: int = Config.WINDOW_SIZE

    # LoRA parameters
    lora_r: int = Config.LORA_R
    lora_alpha: int = Config.LORA_ALPHA
    lora_dropout: float = Config.LORA_DROPOUT
    lora_target_modules: list = field(
        default_factory=lambda: Config.LORA_TARGET_MODULES
    )

    # Hardware
    use_fp16: bool = Config.USE_FP16
    use_8bit: bool = Config.USE_8BIT


class TokenizedWhatsAppDataset(torch.utils.data.Dataset):
    """Wraps WhatsAppDataset with tokenization."""

    def __init__(self, whatsapp_dataset, tokenizer, personas, max_length):
        self.dataset = whatsapp_dataset
        self.tokenizer = tokenizer
        self.personas = personas
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt, completion = self.dataset[idx]

        # Format instruction
        full_text = self._format_instruction(prompt, completion)

        # Tokenize
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze().clone(),
        }

    def _format_instruction(self, prompt: str, completion: str) -> str:
        """Format prompt + completion as instruction."""
        completion_line = completion.strip()

        if "[" in completion_line and "]" in completion_line:
            target_persona = completion_line.split("]")[0].replace("[", "").strip()
            message_text = completion_line.split(":", 2)[-1].strip()
        else:
            target_persona = "Unknown"
            message_text = completion_line

        persona_info = self.personas.get(target_persona, {})
        avg_words = persona_info.get("avg_words", 10)

        instruction = f"""### WhatsApp Conversation:
{prompt.strip()}

### Instruction:
Generate the next message from {target_persona}. Mimic their typical style (avg {avg_words:.0f} words).

### Response from {target_persona}:
{message_text}"""

        return instruction


class WhatsAppTrainer:
    """Handles training of the WhatsApp conversation model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load persona information
        with open(config.personas_json, "r", encoding="utf-8") as f:
            self.personas = json.load(f)

        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self):
        """Create PyTorch dataset (truly on-the-fly)."""
        logger.info("Preparing dataset...")

        # Your efficient WhatsApp dataset
        raw_dataset = WhatsAppDataset(
            self.config.messages_jsonl, window_size=self.config.window_size
        )

        logger.info(f"Created {len(raw_dataset)} training examples")

        # Wrap with tokenization
        tokenized_dataset = TokenizedWhatsAppDataset(
            raw_dataset, self.tokenizer, self.personas, self.config.max_seq_length
        )

        return tokenized_dataset

    def train(self):
        """Train the model using LoRA."""
        logger.info("Starting training...")

        # Prepare dataset
        full_dataset = self.prepare_dataset()

        # Manual train/eval split (90/10)
        train_size = int(0.9 * len(full_dataset))
        eval_size = len(full_dataset) - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, eval_size]
        )

        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        # Load model
        logger.info(f"Loading model {self.config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=20,
            save_steps=50,
            save_total_limit=2,
            report_to="none",
            warmup_steps=10,
            weight_decay=0.01,
            remove_unused_columns=False,  # Important!
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train!
        logger.info("Starting training...")
        trainer.train()

        # Save model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Save personas
        personas_path = os.path.join(self.config.output_dir, "personas.json")
        with open(personas_path, "w", encoding="utf-8") as f:
            json.dump(self.personas, f, indent=2, ensure_ascii=False)

        logger.info("Training complete!")


if __name__ == "__main__":
    config = ModelConfig()
    trainer = WhatsAppTrainer(config)
    trainer.train()
