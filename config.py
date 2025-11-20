"""
Configuration file for WhatsApp Mocker.
Adjust these settings based on your needs and hardware.
"""

def is_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


class BaseConfig:
    """Base configuration - shared across all models."""

    # === Data Paths ===
    RAW_CHAT_DIR = "data"
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "D:\\WhatsappMocker\\models"

    # === Generation Settings ===
    TEMPERATURE = 0.8
    TOP_P = 0.9
    MAX_NEW_TOKENS = 100

    # === Conversation Settings ===
    MIN_MESSAGES_FOR_PERSONA = 5
    DEFAULT_CONVERSATION_TURNS = 10
    # Training feature flags (can be overridden per model)
    GRADIENT_CHECKPOINTING = False


class GPT2Config(BaseConfig):
    """GPT-2 configuration - lightweight, CPU-friendly - optimized for GTX 960M"""

    BASE_MODEL = "gpt2"
    MODEL_OUTPUT_DIR = f"{BaseConfig.MODEL_DIR}/gpt2"

    # Training (auto-adjust for Colab T4 vs local GTX 960M)
    NUM_EPOCHS = 5 if is_colab() else 1
    BATCH_SIZE = 32 if is_colab() else 5
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-4
    LR_SCHEDULER_TYPE = "cosine" if is_colab() else "constant_with_warmup"
    WARMUP_RATIO = 0.1 if is_colab() else 0.05
    MAX_SEQ_LENGTH = 256
    WINDOW_SIZE = 6
    GRADIENT_CHECKPOINTING = True

    # LoRA (bigger capacity for Colab)
    LORA_R = 16 if is_colab() else 8
    LORA_ALPHA = 32 if is_colab() else 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]

    # Hardware
    # Try FP16 to reduce memory; if you see instability/NaNs on older GPUs,
    # set this back to False.
    USE_FP16 = True
    USE_8BIT = False


class Phi2Config(BaseConfig):
    """Phi-2 configuration - balanced quality/speed."""

    BASE_MODEL = "microsoft/phi-2"
    MODEL_OUTPUT_DIR = f"{BaseConfig.MODEL_DIR}/phi2"

    # Training
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    WINDOW_SIZE = 8
    GRADIENT_CHECKPOINTING = True

    # LoRA
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "dense"]  # Phi-2 specific

    # Hardware
    USE_FP16 = True
    USE_8BIT = False  # Set True if GPU < 8GB


class MistralConfig(BaseConfig):
    """Mistral-7B configuration - highest quality."""

    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    MODEL_OUTPUT_DIR = f"{BaseConfig.MODEL_DIR}/mistral"

    # Training
    NUM_EPOCHS = 3
    BATCH_SIZE = 2  # Smaller for large model
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 1024
    WINDOW_SIZE = 12
    GRADIENT_CHECKPOINTING = True

    # LoRA
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Llama-style

    # Hardware
    USE_FP16 = True
    USE_8BIT = True  # Usually needed for 7B models


# Default config - easy to switch
Config = GPT2Config  # ← Change this line to switch models
