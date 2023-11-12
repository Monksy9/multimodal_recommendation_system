import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def get_image_caption(self, img_path: str, conditional_text: str = None) -> str:
        try:
            raw_image = Image.open(img_path)
        except IOError:
            return "Error: The provided file path is not a valid image."

        inputs = self.processor(raw_image, conditional_text, return_tensors="pt") if conditional_text else self.processor(raw_image, return_tensors="pt")
        
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

def main():
    img_path = os.path.join("images", "0237347017.jpg")
    captioner = ImageCaptioner()
    
    caption = captioner.get_image_caption(img_path)
    print(caption)

if __name__ == '__main__':
    main()
