class SimpleTokenizer:
    def __init__(self):
        # Define basic special tokens commonly used in LLMs
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[START]': 2,
            '[END]': 3
        }
        
        # Initialize vocabulary with special tokens
        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.reverse_vocab = {idx: token for token, idx in self.special_tokens.items()}
        
    def tokenize(self, text):
        """
        Basic tokenization of input text into words and subwords
        """
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Basic cleaning
        text = ''.join(c if c.isalnum() or c.isspace() else f' {c} ' for c in text)
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def encode(self, text):
        """
        Convert text to token IDs
        """
        tokens = self.tokenize(text)
        
        # Add tokens to vocabulary if not present
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
        
        # Convert tokens to IDs
        token_ids = [self.vocab.get(token, self.special_tokens['[UNK]']) for token in tokens]
        
        return token_ids
    
    def decode(self, token_ids):
        """
        Convert token IDs back to text
        """
        tokens = [self.reverse_vocab.get(idx, '[UNK]') for idx in token_ids]
        return ' '.join(tokens)


    def subword_tokenize(self, word, max_subword_length=3):
      """
      Break words into subwords if they're too long
      """
      if len(word) <= max_subword_length:
          return [word]
      
      subwords = []
      for i in range(0, len(word), max_subword_length):
          subwords.append(word[i:i + max_subword_length])
      return subwords

# Example usage
def main():
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Example text
    text = "Hello! This is a simple LLM tokenizer test."
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)
    
    # Encode the text to token IDs
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    
    # Decode back to text
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text:", decoded_text)

if __name__ == "__main__":
    main()
