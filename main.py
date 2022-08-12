import cv2

from api import inference_pipeline


# Start here!
username = 'bettersolartester@gmail.com'

# Run module through pipeline.
module = cv2.imread('images/example_module.jpg')
module_shape = (6, 10)
# inference_pipeline(module, username, module_shape, img_type='module')

# Run cell through pipeline.
cell = cv2.imread('images/example_cell.jpg')
inference_pipeline(cell, username, img_type='cell')
