import cv2

from api import inference_pipeline, display_output


# Start here!
username = 'bettersolartester@gmail.com'

# Run module through pipeline.
module = cv2.imread('images/example_module.jpg')
module_shape = (6, 10)
module_output = inference_pipeline(module, username, module_shape, img_type='module')
display_output(module, module_output, module_shape, img_type='module')

# Run cell through pipeline.
cell = cv2.imread('images/example_cell.jpg')
cell_output = inference_pipeline(cell, username, img_type='cell')
display_output(cell, cell_output, img_type='cell')
