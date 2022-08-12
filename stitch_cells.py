import numpy as np 


# Helper function to merge cells to form a module.
def merge_images(image1, image2, type='cell', side=0):
    (height1, width1) = image1.shape[:2]
    (height2, width2) = image2.shape[:2]

    result_height = max(height1, height2) if side == 1 else height1 + height2
    result_width = width1 + width2 if side == 1 else max(width1, width2)

    if type == 'cell':
        result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
        result[:height1, :width1, :] = image1
        if side == 1:
            result[:height2, width1:width1 + width2, :] = image2
        else:
            result[height1:height1 + height2, :width2, :] = image2
    else:
        result = np.zeros((result_height, result_width))
        result[:height1, :width1] = image1
        if side == 1:
            result[:height2, width1:width1 + width2] = image2
        else:
            result[height1:height1 + height2, :width2] = image2
    return result


# Function that takes in a list of cells and their corresponding segmentations, then stitches them together. 
def stitch_cells(cells, processed_cells, height, width):
    rebuild = 1
    height_arr = []
    pro_height_arr = []
    old_col = np.zeros((0, 0, 0), dtype=float)
    old_row = np.zeros((0, 0, 0), dtype=float)
    pro_old_col = np.zeros((0, 0))
    pro_old_row = np.zeros((0, 0))
    for w in range(width):
        for h in range(height):
            idx = width*h + w
            height_arr.append(cells[idx])
            pro_height_arr.append(processed_cells[idx])
        for h in range(height - 1):
            if h == 0:
                col = merge_images(height_arr[h], height_arr[h + 1], side=0)
                pro_col = merge_images(pro_height_arr[h], pro_height_arr[h + 1], type='seg', side=0)
            else:
                col = merge_images(col, height_arr[h + 1], side=0)
                pro_col = merge_images(pro_col, pro_height_arr[h + 1], type='seg', side=0)
            rebuild += 1
        height_arr = []
        pro_height_arr = []
        if w != 0:
            old_col = merge_images(old_col, col, side=1)
            pro_old_col = merge_images(pro_old_col, pro_col, type='seg', side=1)
        else:
            old_col = col
            pro_old_col = pro_col

    for h in range(height):
        for w in range(width):
            idx = width*h + w
            height_arr.append(cells[idx])
            pro_height_arr.append(processed_cells[idx])
        for w in range(width - 1):
            if w == 0:
                row = merge_images(height_arr[w], height_arr[w + 1], side=1)
                pro_row = merge_images(pro_height_arr[w], pro_height_arr[w + 1], type='seg', side=1)
            else:
                row = merge_images(row, height_arr[w + 1], side=1)
                pro_row = merge_images(pro_row, pro_height_arr[w + 1], type='seg', side=1)
        height_arr = []
        pro_height_arr = []
        if h != 0:
            old_row = merge_images(old_row, row, side=0)
            pro_old_row = merge_images(pro_old_row, pro_row, type='seg', side=0)
        else:
            old_row = row
            pro_old_row = pro_row

    return old_col, old_row, pro_old_col, pro_old_row