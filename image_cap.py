import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# loads the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# load your image
img_path = "mushroomhouse.jpg"
# convert it to an RGB format
my_image = Image.open(img_path).convert('RGB')

my_text = "the image of"
inputs = processor(images=my_image, text=my_text, return_tensors="pt")

# generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# print the caption
print(caption)


