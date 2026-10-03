import torch
import time
from diffusers import StableDiffusionPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Device :",device)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to(device)

prompt = input("Enter your text prompt: ")

steps = int(input("Enter number of inference steps : "))

start_time = time.time()

result = pipeline(
    prompt,
    num_inference_steps=steps
)

image = result.images[0]

image.save("output.png")

end_time = time.time()
generation_time = end_time - start_time

print("\n------Image Generation Completed------")
print("prompt :", prompt)
print("Inference steps :", steps)
print("Image saved as output.png")
print("Time taken for generation :", generation_time,"seconds")
print("Device :",device)