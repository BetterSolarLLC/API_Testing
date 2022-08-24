import requests
import json
from io import BytesIO
import cv2
import numpy as np
from stitch_cells import stitch_cells
import matplotlib.pyplot as plt
import matplotlib


# Fill in with current API_IP.
API_IP = ''

# Set colors for custom colormap for image visualizations.
COLOR_LIST = [(0.001462, 0.000466, 0.013866, 1.0),
              (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
              (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
              (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
              (1.0, 0.4980392156862745, 0.0, 1.0)]

# Create the new map.
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', COLOR_LIST, len(COLOR_LIST))

# Used for legend.
DEFECTS = ['None', 'Crack', 'Contact', 'Interconnect', 'Corrosion']
PATCHES = [matplotlib.patches.Patch(color=COLOR_LIST[i], label=DEFECTS[i]) for i in range(1, len(COLOR_LIST))]


# Pipeline for module processing.
def inference_pipeline(image, username, module_shape=None, img_type='module', model_choice='defects+corrosion.pth', threshold=0.52):
    """ Use this function to process images through the model.
    
    Parameters
    ----------
    image : ndarray / cv2 compatible array
        Image to process
    username : str
        Email of user account
    module_shape : int tuple (None if cell)
        Height, Width of module 
    img_type : str
        Either 'module' or 'cell'
    model_choice : str
        Reference to inference model
    threshold : float
        Custom threshold of model confidence (higher = stricter defect threshold)

    Returns
    -------
    model_output : json/str
        status - Success/Failure message
        response : json
            total : total defect percentage
            crack : cracked percentage
            contact : contact defect percentage
            interconnect : interconnect defect percentage
            corrosion/brightspot : defect percentage
            rating : PASS/FAIL (arbitrary)
            segmentation : overlay of segmentation defects
    """
    # Necessary metadata to send with the request.
    meta_data = json.dumps({
        'username': username,
        'model_name': model_choice,
        'threshold': threshold,
        'type': img_type,
        'module_shape': module_shape
    })
    # Allows for sending of larger images.
    if image.shape[0] > 5000 or image.shape[1] > 5000:
        image = cv2.resize(image, [image.shape[1]//2, image.shape[0]//2])
    with BytesIO() as buf:
        buf.write(cv2.imencode('.jpeg', image)[1])
        model_output = requests.post(url=f'{API_IP}/image_inference', files={'image': buf.getvalue(), 'json_data': ('filename', meta_data, 'application/json')})
        try:
            model_output = model_output.json()
        except requests.JSONDecodeError:
            # Modify to handle error differently.
            print(model_output.text)
            return
    
    return model_output


def display_output(image, model_output, module_shape=None, img_type='module'):
    # Above is all you need for the request. Below is combining everything for print and display.
    if img_type == 'module':
        total_defective_area = np.zeros(4)
        # module = np.array(model_output['module'])
        cells = []
        processed_cells = []
        for cell_output in model_output['cell_outputs']:
            # Add up total defective area of all cells.
            total_defective_area += [cell_output['crack'], cell_output['corrosion'], cell_output['interconnect'], cell_output['corrosion']]
            processed_cells.append(np.array(cell_output['segmentation']))
        for cell in model_output['cells']:
            cells.append(np.array(cell))
        
        h, w = module_shape
        # Stitch cells into module. (try row if col doesn't work well)
        stitch_col, stitch_row, seg_stich_col, seg_stitch_row = stitch_cells(cells, processed_cells, h, w)

        module_image, segmentation = stitch_col, seg_stich_col
        stats = total_defective_area / len(cells)
        segmentation_mask = np.ma.masked_where(segmentation == 0, segmentation)
        fig, ax = plt.subplots()
        ax.imshow(module_image, cmap='gray', vmin=0, vmax=1)
        ax.imshow(segmentation_mask, cmap=CMAP, vmin=0, vmax=4, alpha=0.3)
        ax.axis('off')
        ax.set_title('example_module')
        ax.legend(handles=PATCHES, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
        plt.show()
        print(f'Cracks: {stats[0]:.4f}%')
        print(f'Contact: {stats[1]:.4f}%')
        print(f'Interconnect: {stats[2]:.4f}%')
        print(f'Corrosion: {stats[3]:.4f}%')
        print(f'Total: {sum(stats):.4f}% defective area.')
    
    if img_type == 'cell':
        segmentation = np.array(model_output['response']['segmentation'])
        segmentation_mask = np.ma.masked_where(segmentation == 0, segmentation)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.imshow(segmentation_mask, cmap=CMAP, vmin=0, vmax=4, alpha=.3)
        ax.axis('off')
        ax.set_title('example_cell')
        ax.legend(handles=PATCHES, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
        plt.show()
        defect_sizes = [model_output['response']['crack'], model_output['response']['contact'],
                        model_output['response']['interconnect'], model_output['response']['corrosion'],
                        model_output['response']['total']]
        print(f'Cracks: {defect_sizes[0]}%')
        print(f'Contact: {defect_sizes[1]}%')
        print(f'Interconnect: {defect_sizes[2]}%')
        print(f'Corrosion: {defect_sizes[3]}%')
        print(f'Total: {defect_sizes[4]}% defective area.')
