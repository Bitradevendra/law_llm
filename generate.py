import torch
import argparse
from model import CheckpointedTransformerBlock

# A simple, placeholder tokenizer.
# In a real application, you would use a library like `tokenizers` or `transformers`.
class SimpleTokenizer:
    def __init__(self):
        # Simple mapping for demonstration purposes
        self.vocab = {chr(i): i for i in range(256)} # Basic ASCII
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(char, 0) for char in text]

    def decode(self, tokens):
        return ''.join([self.inverse_vocab.get(token, '') for token in tokens])

def generate_response(model, tokenizer, prompt, max_length=50, device='cuda'):
    """Generates a response from the model given a prompt."""
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            # The model expects a batch dimension, so we keep it.
            output = model(input_tensor)
            
            # Get the predicted next token (we take the last token's logits)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append the predicted token to the input for the next step
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Stop if the model generates an end-of-sequence token (if any)
            # For this simple case, we just append the token value.
            token_item = next_token.item()
            generated_tokens.append(token_item)

    return tokenizer.decode(generated_tokens)

def main():
    parser = argparse.ArgumentParser(description="Interact with a trained LLM.")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--dim", type=int, default=768, help="Model dimension.")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model and Tokenizer
    model = CheckpointedTransformerBlock(dim=args.dim, num_heads=args.heads).to(device)
    tokenizer = SimpleTokenizer()

    # 2. Load the trained weights
    try:
        # The checkpoint saved by DeepSpeed/DDP might be nested under 'module'
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Remove the 'module.' prefix if it exists
        unwrapped_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(unwrapped_state_dict)
        print("Model checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 3. Interaction Loop
    print("\nLLM is ready. Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
        
        response = generate_response(model, tokenizer, prompt, device=device)
        print(f"LLM: {response}")

if __name__ == "__main__":
    main()
