import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained Stable Diffusion model (Text-to-Image Generation)
from huggingface_hub import login

# Login to Hugging Face (use your API token here)
login(token="<USE YOUR API TOKEN HERE>")

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", 
    torch_dtype=torch.float16
)

pipe.to("cuda")  # Move model to GPU

# Load the pre-trained CLIP model (Text-Image Matching)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Generate Image from Text
def generate_image_from_text(prompt):
    image = pipe(prompt).images[0]
    return image

# Evaluate Image-Text Similarity using CLIP
def evaluate_image_text_similarity(text, image):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.cosine_similarity(text_features, image_features)
    return similarity.item()

# Main function to generate image and evaluate similarity
def main():
    # Text prompt to generate an image
    prompt = "A futuristic city with flying cars"

    # Step 1: Generate Image from Text
    print("Generating image from text prompt...")
    generated_image = generate_image_from_text(prompt)
    generated_image.save("generated_image.png")
    generated_image.show()

    # Step 2: Evaluate the similarity of the generated image to the text
    print("Evaluating image-text similarity...")
    similarity_score = evaluate_image_text_similarity(prompt, generated_image)
    print(f"Similarity between text and generated image: {similarity_score:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
