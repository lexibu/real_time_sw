import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# URL of the page
url = "https://moon.nasa.gov/moon-observation/daily-moon-guide/?intent=011#1723596157065::0::"

# Make a request to fetch the HTML content of the page
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Find the image URL
# You may need to inspect the page to find the correct image selector
# Here, we'll look for an <img> tag with a specific class or identifier
img_tag = soup.find(
    "img", {"class": "some-class"}
)  # Replace 'some-class' with actual class or identifier
if img_tag:
    img_url = img_tag["src"]
else:
    raise ValueError("Image URL not found")

# Make a request to fetch the image
img_response = requests.get(img_url)
img = Image.open(BytesIO(img_response.content))

# Save the image as PNG
img.save("moon_image.png")

print("Image saved as 'moon_image.png'")
