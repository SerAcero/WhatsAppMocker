"""
Train an LLM to mimic WhatsApp conversation styles.
Uses LoRA fine-tuning for efficient training on consumer hardware.
"""

import json
import os
from dataclasses import dataclass, field

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
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
    lr_scheduler_type: str = Config.LR_SCHEDULER_TYPE
    warmup_ratio: float = Config.WARMUP_RATIO
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
    gradient_checkpointing: bool = Config.GRADIENT_CHECKPOINTING


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
        full_text = format_instruction(prompt, completion, self.personas)

        full_text = full_text + self.tokenizer.eos_token
        # print(full_text)

        # Compute prefix length (prompt + "### Response from ...:\n") to ignore in loss
        response_header = "### Response from"
        header_pos = full_text.find(response_header)
        if header_pos != -1:
            newline_after_header = full_text.find("\n", header_pos)
            split_idx = (
                newline_after_header + 1
                if newline_after_header != -1
                else len(full_text)
            )
        else:
            split_idx = len(full_text)
            logger.warning(
                f"Missing response header in sample {idx}, masking entire text"
            )
        prefix_text = full_text[:split_idx]

        # Tokenize full sample (padded) and prefix (no padding)
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prefix_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prefix_len = min(len(prefix_ids), self.max_length)

        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        real_len = int(attention_mask.sum().item())

        labels = input_ids.clone()
        # Ignore prompt/header tokens
        labels[:prefix_len] = -100
        # Ignore padded positions
        labels[attention_mask == 0] = -100

        # Fallback: ensure at least one supervised token to avoid NaN loss
        if (labels != -100).sum().item() == 0 and real_len > 0:
            last_idx = real_len - 1  # last non-padding token
            labels[last_idx] = input_ids[last_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def analyze(self):
        """Analyze dataset token lengths and masking statistics."""
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)

        total_samples = len(self)
        all_masked_count = 0
        total_tokens = 0
        total_supervised = 0
        total_ignored = 0

        supervised_tokens_list = []
        ignored_tokens_list = []

        for i in range(total_samples):
            sample = self[i]
            labels = sample["labels"]
            attention_mask = sample["attention_mask"]

            # Count real tokens (non-padding)
            real_tokens = int(attention_mask.sum().item())
            # Count supervised tokens (not -100)
            supervised = int((labels != -100).sum().item())
            # Count ignored tokens (prompt + padding)
            ignored = real_tokens - supervised

            total_tokens += real_tokens
            total_supervised += supervised
            total_ignored += ignored

            supervised_tokens_list.append(supervised)
            ignored_tokens_list.append(ignored)

            if supervised == 0:
                all_masked_count += 1

        print(f"Total samples: {total_samples}")
        print(
            f"Samples with ALL tokens masked: {all_masked_count} ({100*all_masked_count/total_samples:.1f}%)"
        )
        print(f"\nAverage tokens per sample: {total_tokens/total_samples:.1f}")
        print(f"Average supervised tokens: {total_supervised/total_samples:.1f}")
        print(f"Average ignored tokens: {total_ignored/total_samples:.1f}")
        print("\nSupervised token stats:")
        print(f"  Min: {min(supervised_tokens_list)}")
        print(f"  Max: {max(supervised_tokens_list)}")
        print(f"  Median: {sorted(supervised_tokens_list)[total_samples//2]}")

        # Show a few examples
        print("\n" + "=" * 60)
        print("SAMPLE EXAMPLES")
        print("=" * 60)

        for i in [0, total_samples // 4, total_samples // 2]:
            sample = self[i]
            real_len = int(sample["attention_mask"].sum().item())
            supervised = int((sample["labels"] != -100).sum().item())

            print(f"\nSample {i}:")
            print(f"  Real tokens: {real_len}")
            print(f"  Supervised tokens: {supervised}")
            print(f"  Ignored tokens: {real_len - supervised}")

            print("  Decoded (first 200 chars):")
            decoded = self.tokenizer.decode(sample["input_ids"][:real_len])
            print(f"    {decoded[:200]}...")

            # Decode the last 200 characters of the response (supervised tokens)
            # Find where supervised tokens start (first non -100 label)
            supervised_start = 0
            for j in range(len(sample["labels"])):
                if sample["labels"][j] != -100:
                    supervised_start = j
                    break

            # Get supervised tokens only
            supervised_ids = sample["input_ids"][supervised_start:real_len]
            decoded_response = self.tokenizer.decode(supervised_ids)

            print("  Decoded response (last 200 chars):")
            print(
                f"    ...{decoded_response[-200:] if len(decoded_response) > 200 else decoded_response}"
            )

        print("\n" + "=" * 60)


def format_instruction(
    prompt: str, completion: str, personas: dict, for_training=True
) -> str:
    """Format prompt + completion as instruction."""
    completion_line = completion.strip()

    if "[" in completion_line and "]" in completion_line:
        target_persona = completion_line.split("]")[0].replace("[", "").strip()
    else:
        target_persona = "Unknown"

    persona_info = personas.get(target_persona, {})
    avg_words = persona_info.get("avg_words", 10)

    instruction = f"""### WhatsApp Conversation:
{prompt.strip()}

### Instruction:
Generate the next message from {target_persona}. Mimic their typical style (avg {avg_words:.0f} words).

### Response from {target_persona}:"""

    if for_training:
        return instruction + f"\n{completion_line}"
    else:
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

    def train(self, profile_only: bool = False, resume_from_checkpoint=False):
        """Train the model using LoRA.

        Args:
            profile_only: If True, only profile first 5 steps and exit without full training.
        """
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

        # Calculate steps per epoch
        effective_batch_size = (
            self.config.batch_size * self.config.gradient_accumulation_steps
        )
        steps_per_epoch = len(train_dataset) // effective_batch_size
        logger.info(
            f"Steps per epoch: {steps_per_epoch} "
            f"(batch_size={self.config.batch_size}, "
            f"grad_accum={self.config.gradient_accumulation_steps})"
        )

        # Set eval/save steps as percentage of epoch (e.g., 20% of epoch)
        logging_percentage = 0.05
        eval_percentage = 0.5
        save_percentage = 0.4

        logging_steps = max(1, int(steps_per_epoch * logging_percentage))
        eval_steps = max(1, int(steps_per_epoch * eval_percentage))
        save_steps = max(1, int(steps_per_epoch * save_percentage))

        logger.info(
            f"Eval every {eval_steps} steps ({eval_percentage*100:.0f}% of epoch), "
            f"Save every {save_steps} steps ({save_percentage*100:.0f}% of epoch)"
        )

        # Load model
        logger.info(f"Loading model {self.config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
        )

        # Enable memory-saving options
        if self.config.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
            # Required for gradient checkpointing with HF models
            if hasattr(model, "config"):
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

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
            max_steps=5 if profile_only else -1,  # -1 means use num_train_epochs
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=max(10, int(steps_per_epoch * self.config.warmup_ratio)),
            # warmup_steps=10,
            logging_steps=logging_steps,
            # Disable eval in profile mode
            # eval_strategy="epoch",
            eval_strategy=("no" if profile_only else "steps"),
            eval_steps=eval_steps if not profile_only else 999999,
            # Don't save during profiling
            save_steps=(save_steps if not profile_only else 999999),
            save_total_limit=2,
            report_to="none",
            weight_decay=0.01,
            remove_unused_columns=False,  # Important!
            fp16=self.config.use_fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            ## PROBABLY WINDOWS SPECIFIC. REMOVE FOR LINUX
            # dataloader_pin_memory=False,  # Skip expensive cudaHostAlloc (Copilot suggestion, I do not know what I am doing)
            # dataloader_num_workers=0,  # No multiprocessing overhead on Windows
            ##
            resume_from_checkpoint=os.path.join(
                self.config.output_dir, "checkpoint-966"
            ),  # Path to your checkpoint
        )

        # print(training_args)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if profile_only:
            # Profile mode: run 5 steps and export trace
            logger.info("PROFILING MODE: Running 5 steps only...")
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                with record_function("model_training"):
                    trainer.train()

            # Prepare output
            output_lines = []
            output_lines.append("=" * 80)
            output_lines.append("PROFILING RESULTS (sorted by CUDA time)")
            output_lines.append("=" * 80)
            output_lines.append(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            )
            output_lines.append("\n" + "=" * 80)
            output_lines.append("PROFILING RESULTS (sorted by CPU time)")
            output_lines.append("=" * 80)
            output_lines.append(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            )

            # Print to console
            print("\n".join(output_lines))

            # Save to file
            profile_output_path = os.path.join(
                self.config.output_dir, "profiling_results.txt"
            )
            with open(profile_output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            logger.info(f"Profiling results saved to {profile_output_path}")

            # Export for viewing in Chrome (chrome://tracing)
            trace_path = os.path.join(self.config.output_dir, "trace.json")
            prof.export_chrome_trace(trace_path)
            logger.info(f"Trace exported to {trace_path}")
            logger.info(
                "Open chrome://tracing in Chrome and load trace.json to visualize"
            )

            return  # Exit without full training

        # Train!
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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

    dataset = trainer.prepare_dataset()
    # dataset.analyze()

    # trainer.train(profile_only=True)

    # sample = dataset[3]
    # print("input_ids:", sample["input_ids"][:])
    # print("attention_mask:", sample["attention_mask"][:])
    # print("labels:   ", sample["labels"][:])
    # print("Decoded:", trainer.tokenizer.decode(sample["input_ids"][:]))

    # print("len(labels):   ", len(sample["labels"]))

    datasetW = dataset.dataset

    prompt, completion = datasetW[0]
    full_text = format_instruction(
        prompt, completion, dataset.personas, for_training=True
    )
    print("\nFULL TEXT:\n", full_text)

    # trainer.train()
#
