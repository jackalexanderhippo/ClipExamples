import torch
import clip
from PIL import Image

# Load the model and preprocess (ViT model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # Using Vision Transformer (ViT-B/32)

# Load an image and preprocess it
image = preprocess(Image.open("your_image.jpg")).unsqueeze(0).to(device)

# Prepare a set of text prompts to compare
texts = clip.tokenize(["a cat", "a dog", "a car"]).to(device)

# Perform forward pass (encode image and text)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate similarity between image and text (cosine similarity)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Display the results
best_match = similarity.argmax().item()
print(f"The image most closely resembles: {['a cat', 'a dog', 'a car'][best_match]}")
