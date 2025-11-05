"""
Configuration file for WhatsApp Mocker.
Adjust these settings based on your needs and hardware.
"""


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


class GPT2Config(BaseConfig):
    """GPT-2 configuration - lightweight, CPU-friendly."""

    BASE_MODEL = "gpt2"
    MODEL_OUTPUT_DIR = f"{BaseConfig.MODEL_DIR}/gpt2"

    # Training
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 256  # GPT-2 works well with shorter context
    WINDOW_SIZE = 6

    # LoRA
    LORA_R = 8  # Smaller for GPT-2
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # GPT-2 specific layers

    # Hardware
    USE_FP16 = False  # Not needed for small model
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
