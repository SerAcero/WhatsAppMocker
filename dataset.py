# whatsapp_dataset.py
import json
from torch.utils.data import Dataset


class WhatsAppDataset(Dataset):
    """
    PyTorch Dataset that reads a WhatsApp conversation (JSONL)
    and yields (prompt, completion) text pairs for LLM training.

    It lazily builds sliding windows over messages so you don't
    create thousands of precomputed examples.
    """

    def __init__(self, jsonl_path: str, window_size: int = 6) -> None:
        """
        Args:
            jsonl_path: Path to messages.jsonl (from parse_whatsapp.py)
            window_size: Number of messages per training example
                         (last message is the target/completion)
        """
        self.window_size = window_size

        # Load all messages once into memory
        self.messages = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.messages.append(json.loads(line))

        if len(self.messages) < window_size:
            raise ValueError(
                f"Not enough messages ({len(self.messages)}) for window {window_size}"
            )

    def __len__(self) -> int:
        """Number of sliding windows available."""
        return len(self.messages) - self.window_size + 1

    def _format_turn(self, msg) -> str:
        """Format one message as a WhatsApp-style line."""
        dt = msg["datetime"]
        ts = dt.split("T")[-1][:5] if "T" in dt else dt
        return f"[{msg['persona_id']}] {ts}: {msg['text']}"

    def __getitem__(self, idx: int):
        """Return (prompt, completion) pair for index `idx`."""
        window = self.messages[idx : idx + self.window_size]

        # Prompt = all turns except last
        prompt = "\n".join(self._format_turn(m) for m in window[:-1]) + "\n"
        # Completion = last turn (the one to predict)
        completion = " " + self._format_turn(window[-1]) + "\n"

        return prompt, completion


if __name__ == "__main__":
    # Example usage
    dataset = WhatsAppDataset("data/processed/messages.jsonl", window_size=6)
    print(f"Dataset has {len(dataset)} examples")

    for i in range(3):
        prompt, completion = dataset[i]
        print(f"\nExample {i}:")
        print("PROMPT:")
        print(prompt)
        print("COMPLETION:")
        print(completion)
