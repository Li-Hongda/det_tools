import os
import glob
src_dir = ''
result_dir = ''
image_dir = glob.glob(os.path.join(src_dir, '*.png'))
for image in image_dir:
    imagename = os.path.basename(image)
    if not imagename in result_dir:
        labelname = imagename.split('.')[0] + '.txt'
        os.system(f"touch {result_dir}/{labelname}") 