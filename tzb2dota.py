import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob



def get_args():
    parser = argparse.ArgumentParser("convert tzb-format to dota-format")
    parser.add_argument("--ori_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    return parser.parse_args()


def tzb2dota(ori_dir, save_dir):
    assert os.path.exists(ori_dir), f"{ori_dir} don't exist"
    os.makedirs(save_dir, exist_ok=True)

    for ann_file in glob(os.path.join(ori_dir, "*.txt")):
        save_ann_file = os.path.join(save_dir, os.path.basename(ann_file))
        assert os.path.exists(ann_file), "ann file exist"

        new_lines = []
        with open(ann_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                new_line = line.strip()[2:] + ' ' + line[0] + ' 0' + ' \n'
                new_lines.append(new_line)

        if len(new_lines):
            with open(save_ann_file, "w") as fw:
                fw.write("imagesource:GoogleEarth\n")
                fw.write("gsd:null\n")
                for line in new_lines:
                    fw.write(line)
                    
def main():
    args = get_args()
    ori_dir = args.ori_dir
    save_dir = args.save_dir
    tzb2dota(ori_dir, save_dir)

if __name__ == "__main__":
    main()