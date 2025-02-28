import PIL
from PIL import Image
import imageio
import numpy as np

filepath = 'assets/ref_images/1.png'
image = Image.open(filepath)
origin_image = image
print(image.size)
height, width, channels = imageio.imread(filepath).shape
print(height, width, channels)


image = np.array(image)
print(image.shape)
height, width = 480, 720
image_height, image_width, _ = image.shape
print(image_height / image_width)
# Fix the height
if image_height / image_width < height / width: # the ratio < 2:3
    print("crop")
    # crop the width
    croped_width = int(image_height / height * width)  
    # 计算目标宽度: target_width = image_height / height * width
    left = (image_width - croped_width) // 2 # 左右两边裁剪
    image = image[:, left: left+croped_width]
else:
    print("pad")
    # pad the width
    pad = int((((width / height) * image_height) - image_width) / 2.)
    padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8)
    padded_image[:, pad:pad+image_width] = image
    image = padded_image

resize_image = Image.fromarray(image).resize((width, height)) # resize
resize_image.show()

# try
origin_image = origin_image.resize((width,height))
origin_image.show()
