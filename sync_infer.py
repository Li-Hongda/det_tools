import os
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path
from queue import Queue

import cv2
from mmdet.apis import inference_detector, init_detector
from mmrotate.structures import rbox2qbox
from mmrotate.utils import register_all_modules


def get_args():
    parser = ArgumentParser()
    parser.add_argument('watch_dir', type=str, default=None)
    parser.add_argument('save_dir', type=str, default=None)

    parser.add_argument('--configs', nargs="+", type=str, default=None, help='Config file')
    parser.add_argument('--ckpts', nargs="+", type=str, default=None, help='Checkpoint file')
    parser.add_argument('--save_name', type=str, default="model")

    parser.add_argument(
        '--device', default='cuda:0', type=str, help='Device used for inference')

    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--interval', type=float, default=0.01)
    args = parser.parse_args()
    return args

def resp_mkdir(dirname: Path, msg: str = None):
    if dirname.exists():
        ans = ""
        while ans not in ["yes", "no"]:
            msg = msg if msg is not None else \
                f"find exists {dirname}, do you want to" \
                " delete it (yes or no): "
            ans = input(msg).lower()
            if ans == "yes":
                shutil.rmtree(dirname)
                dirname.mkdir()
            elif ans == "no":
                exit(0)
    else:
        dirname.mkdir()

def set_results(result, save_dir):
    img_path = result.img_path
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    bboxes = result.pred_instances.bboxes
    bboxes = rbox2qbox(bboxes).detach().cpu().numpy().tolist()
    scores = result.pred_instances.scores.detach().cpu().numpy().tolist()
    labels = result.pred_instances.labels.detach().cpu().numpy().tolist()
    save_file = os.path.join(save_dir, img_name + ".txt")

    lines = ""
    for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
        bbox_str = list(map(str, bbox))
        bbox_str.append(str(label))
        bbox_str.append(str(score))
        line =f'{i}' + " "+ " ".join(bbox_str)
        if i != len(labels) - 1:
            line += "\n"
        lines += line
    with open(save_file, "w") as f:
        f.write(lines)

def main():
    register_all_modules()

    args = get_args()    
    watch_dir = args.watch_dir
    assert os.path.exists(watch_dir), f"{watch_dir} don't exists"
    save_dir = args.save_dir
    save_name = args.save_name
    interval = args.interval
    resp_mkdir(Path(save_dir))

    configs = args.configs
    checkpoints = args.ckpts

    models = []
    save_dirs = []
    for i, (config, checkpoint) in enumerate(zip(configs, checkpoints)):
        model = init_detector(
            config, checkpoint, 'none', device=args.device)
        models.append(model)
        name = save_name + str(i) 
        sub_save_dir = os.path.join(save_dir, name)
        save_dirs.append(sub_save_dir)
        os.makedirs(sub_save_dir)
   
    que = Queue(maxsize=0) 
    old_files_set = []
    new_files_set = []
    while True:
        new_files_set = os.listdir(watch_dir)
        if len(new_files_set) > len(old_files_set):
            add_set = set(new_files_set) ^ set(old_files_set)
        else:
            add_set = []

        print(f"total: {len(new_files_set)} | queue size {que.qsize()} | add {len(add_set)}")
        old_files_set = new_files_set
        new_files_set = []

        for add in add_set:
            que.put(add)

        if not que.empty():
            img_name = que.get()
            img_file = os.path.join(watch_dir, img_name)
            img = None
            while img is None:
                try:
                    img = cv2.imread(img_file)
                except:
                    img = None
                    time.sleep(0.5)

            for i, model in enumerate(models):
                result = inference_detector(model, img_file)
                set_results(result, save_dirs[i])

        time.sleep(interval)
    

if __name__ == "__main__":
    main()