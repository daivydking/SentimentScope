# SentimentScope: Sentiment Analysis with Transformers

A transformer model trained from scratch in PyTorch to classify IMDB movie reviews as positive or negative, built for CineScope's recommendation system.

## Results
- Test accuracy: **75.77%** (target: above 75%)
- Validation accuracy: 72.64% after epoch 1, 77.68% after epoch 3

## Approach
- **Data:** IMDB dataset (25,000 train / 25,000 test reviews), split 90/10 into training and validation sets. HTML `<br />` tags removed.
- **Tokenization:** `bert-base-uncased` subword tokenizer, padded or truncated to 128 tokens.
- **Model:** 4 transformer blocks, 4 attention heads, embedding size 128, mean pooling, and a linear classification head with 2 outputs.
- **Training:** cross-entropy loss, AdamW (learning rate 3e-4), batch size 32, 3 epochs.

## Files
- `SentimentScope_starter.ipynb`: the full notebook with outputs
- `model_checkpoint.pth`: trained model weights

## Load the checkpoint
```python
model = DemoGPT(config)
model.load_state_dict(torch.load("model_checkpoint.pth"))
model.eval()
```
