"""
Generate WhatsApp-style responses using the trained model.
Can simulate multi-person conversations.
"""

import json
import os
import random
from typing import List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config
from train_model import format_instruction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhatsAppGenerator:
    """Generates WhatsApp-style messages mimicking different personas."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Args:
            model_path: Path to trained LoRA model (default: from Config)
            temperature: Sampling temperature (default: from Config)
            top_p: Nucleus sampling parameter (default: from Config)
            max_new_tokens: Maximum tokens to generate (default: from Config)
        """
        # Use config defaults if not specified
        self.model_path = model_path or Config.MODEL_OUTPUT_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generation parameters - use Config defaults
        self.temperature = temperature or Config.TEMPERATURE
        self.top_p = top_p or Config.TOP_P
        self.max_new_tokens = max_new_tokens or Config.MAX_NEW_TOKENS
        self.repetition_penalty = Config.REPETITION_PENALTY
        # self.no_repeat_ngram_size = Config.NO_REPEAT_NGRAM_SIZE

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model path: {self.model_path}")

        # Load personas
        personas_path = os.path.join(self.model_path, "personas.json")
        if not os.path.exists(personas_path):
            raise FileNotFoundError(f"Personas file not found: {personas_path}")

        with open(personas_path, "r", encoding="utf-8") as f:
            self.personas = json.load(f)

        logger.info(f"Loaded {len(self.personas)} personas")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect base model if not provided
        config_path = os.path.join(self.model_path, "adapter_config.json")
        base_model_name = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")

        # Fallback to Config if not found in adapter_config
        if base_model_name is None:
            base_model_name = Config.BASE_MODEL
            logger.warning(
                f"Could not find base model in adapter_config.json, "
                f"using Config.BASE_MODEL: {base_model_name}"
            )

        logger.info(f"Loading base model: {base_model_name}")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Load LoRA adapters on top of base model
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()  # Set to evaluation mode
        self.model.to(self.device)

        logger.info(f"Model loaded successfully on {self.device}")

    def generate_response(
        self,
        conversation_history: List[str],
        target_persona: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from target_persona given conversation history.

        Args:
            conversation_history: List of recent messages in format:
                ["[Alice] Hey, how are you?", "[Bob] Good! You?", ...]
            target_persona: Name of persona to generate response for (e.g., "Alice")
            temperature: Sampling temperature (default: from config)
            top_p: Nucleus sampling (default: from config)
            max_new_tokens: Max tokens to generate (default: from config)

        Returns:
            Generated message text (without persona prefix)
        """
        # Use instance defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )

        # Keep only last window_size messages for context
        if len(conversation_history) > Config.WINDOW_SIZE:
            conversation_history = conversation_history[-Config.WINDOW_SIZE :]

        # Format prompt exactly as training
        history_text = "\n".join(conversation_history)
        dummy_completion = f"[{target_persona}] "  # Placeholder
        prompt = format_instruction(
            history_text, dummy_completion, self.personas, for_training=False
        )
        prompt += f"\n[{target_persona}]:"
        # print(f"Formatted prompt:\n{prompt}\n{'-'*40}\n")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.repetition_penalty,
                # no_repeat_ngram_size=self.no_repeat_ngram_size,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full generated text: {generated_text}\n\n")

        # Extract only the response part (after the response marker)
        response_marker = f"### Response from {target_persona}:"
        if response_marker in generated_text:
            response = generated_text.split(response_marker)[-1].strip()
            print(f"Generated response: {response}\n\n")

            if len(response) == 0:
                logger.warning("Generated response is empty after extraction.")
                return f"[{target_persona}]: !! No generé respuesta"
            return response

        else:
            # Fallback: return everything after the prompt
            logger.warning(
                f"Response marker not found in generation: {generated_text[:100]}..."
            )
            return generated_text[len(prompt) :].strip()

    def _extract_response(self, full_text: str, prompt: str) -> str:
        """Extract the generated response from full text."""
        # Remove the prompt
        response = full_text[len(prompt) :].strip()

        # Stop at newline or end markers
        for stop in ["\n\n", "###", "Instruction:", "Response from"]:
            if stop in response:
                response = response.split(stop)[0]

        return response.strip()

    def simulate_conversation(
        self,
        conversation: List[str],
        num_turns: int = 10,
        active_personas: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Simulate a multi-person conversation.

        Args:
            initial_message: {"persona": "Name", "text": "message"}
            num_turns: Number of message exchanges
            active_personas: List of personas who might respond (None = all)

        Returns:
            Full conversation history
        """

        # Filter personas (exclude SYSTEM)
        available_personas = [
            p for p in self.personas.keys() if self.personas[p]["n_messages"] > 5
        ]

        if active_personas:
            available_personas = [p for p in available_personas if p in active_personas]

        logger.info(f"Simulating conversation with personas: {available_personas}")

        for turn in range(num_turns):
            # Choose next persona (not the last one to speak)
            # last_speaker = conversation[-1]["persona"]
            # candidates = [p for p in available_personas if p != last_speaker]
            candidates = available_personas.copy()

            if not candidates:
                break

            # Weight by message frequency
            weights = [self.personas[p]["n_messages"] for p in candidates]
            next_persona = random.choices(candidates, weights=weights)[0]

            # Generate response
            logger.info(f"Turn {turn + 1}: {next_persona} responding...")
            response = self.generate_response(conversation, next_persona)

            # Add to conversation
            # conversation.append(f"[{next_persona}]: {response}")
            conversation.append(f"{response}")

            # print(f"{next_persona}: {response}")

        return conversation

    def get_available_personas(self) -> List[str]:
        """Get list of available personas."""
        return [p for p in self.personas.keys() if p != "SYSTEM"]


def interactive_mode(model_path: Optional[str] = None):
    """Interactive chat mode."""
    print("=" * 60)
    print("WhatsApp Conversation Simulator")
    print("=" * 60)

    # Initialize generator
    generator = WhatsAppGenerator(model_path=model_path)

    # Show available personas
    personas = generator.get_available_personas()
    print(f"\nAvailable personas: {', '.join(personas)}")

    # Get user choice
    print("\nChoose mode:")
    print("1. Chat as yourself (others will respond)")
    print("2. Simulate autonomous conversation")

    # choice = input("\nEnter choice (1/2): ").strip()
    choice = 2

    if choice == "1":
        # User participates
        user_name = input("Your name: ").strip()

        conversation = []
        print("\nStart chatting! (type 'quit' to exit)\n")

        while True:
            # User input
            user_msg = input(f"{user_name}: ").strip()

            if user_msg.lower() in ["quit", "exit", "q"]:
                break

            conversation.append({f"[{user_name}]: {user_msg}"})

            # Random persona responds
            candidates = [
                p
                for p in personas
                if len(conversation) < 2 or p != conversation[-2].get("persona")
            ]
            responder = random.choice(candidates)

            responses = generator.generate_response(conversation, responder)
            response = responses[0]

            conversation.append({f"[{responder}]: {response}"})

            print(f"[{responder}]: {response}\n")

    else:
        # Autonomous simulation

        # print(f"\nAvailable personas: {', '.join(personas)}")
        # persona_input = input("Initial persona (press Enter for random): ").strip()
        persona_input = "Sergio"

        if persona_input and persona_input in personas:
            initial_persona = persona_input
        elif persona_input:
            print(f"Warning: '{persona_input}' not found. Using random persona.")
            initial_persona = random.choice(personas)
        else:
            initial_persona = random.choice(personas)

        # initial_msg = input("\nInitial message: ").strip()
        initial_msg = "Alguien algo?"

        # num_turns = int(input("Number of turns (default 10): ").strip() or "10")
        num_turns = 40

        print(f"\n[{initial_persona}]: {initial_msg}\n")

        conversation = generator.simulate_conversation(
            [f"[{initial_persona}]: {initial_msg}"], num_turns=num_turns
        )

        print("\n" + "=" * 60)
        # Save to file
        conversation_output_path = os.path.join(
            Config.MODEL_OUTPUT_DIR, "conversation_results.txt"
        )
        with open(conversation_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(conversation))
        logger.info(f"Conversation results saved to {conversation_output_path}")
        print("Conversation saved!")


if __name__ == "__main__":
    interactive_mode()
    # WG = WhatsAppGenerator()

    # conversation_history = [
    #     "[Juan V.] 10:00: Qué pasa, tíos?",
    #     "[Edu] 10:01: Yo hoy no hago na",
    # ]
    # target_persona = "Juan V."
    # WG.generate_response(conversation_history, target_persona)
