# Groq - Vision

**Pages:** 2

---

## Function to encode the image

**URL:** llms-txt#function-to-encode-the-image

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

---

## Path to your image

**URL:** llms-txt#path-to-your-image

image_path = "sf.jpg"

---
