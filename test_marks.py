from PIL import Image, ImageDraw
from backend.screen_capture import ScreenCapture

# Create a dummy image
img = Image.new('RGB', (800, 600), color = (73, 109, 137))
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 200, 200], fill=(0, 255, 0))
draw.rectangle([300, 100, 400, 200], fill=(0, 0, 255))
draw.rectangle([500, 300, 700, 400], fill=(255, 0, 0))

marked_img, id_map = ScreenCapture.draw_set_of_marks(img)
print(id_map)
