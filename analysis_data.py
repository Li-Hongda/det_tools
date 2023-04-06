import os
import mmengine
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import argparse
import mmdet

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis data')
    parser.add_argument("--task", default='category', type=str)
    parser.add_argument("--label_dir", type=str, default='')
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--label-type", default= '.txt')
    parser.add_argument("--out", default='work_dirs/analysis/')
    args = parser.parse_args()
    return args


def plot_bar(cate_dict, args):
    """Plot bar."""
    categories = list(cate_dict.keys())
    categories.sort()
    xs = []
    ys = []
    for i in range(len(categories)):
        xs += [categories[i]]
        ys += [cate_dict[categories[i]]]
        plt.xlabel('category')
        plt.ylabel('number')
        plt.bar(xs, ys)

        plt.legend()
    if args.out is None:
        plt.show()
    else:
        print(f'save bar to: {args.out}')
        plt.savefig(os.path.join(args.out, f'{args.task}.png'))
        plt.cla()


def plot_scatter(x_list, y_list, args):
    plt.scatter(x_list, y_list, s = 10)
    plt.xticks(np.arange(0,240)[::20])
    plt.yticks(np.arange(0,220)[::20])
    if args.out is None:
        plt.show()
    else:
        print(f'save scatter to: {args.out}')
        plt.savefig(os.path.join(args.out, f'{args.task}.png'))
        plt.cla()
        

def analysis_category(args):
    if args.label_type == '.json':
        pass
    categories = dict()
    file_generator = mmengine.scandir(args.label_dir)
    for file in mmengine.track_iter_progress(list(file_generator)):
        with open(os.path.join(args.label_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splitline = line.split(' ')
                if splitline[0] not in categories:
                    categories[splitline[0]] = 1
                else:
                    categories[splitline[0]] += 1
    with open(os.path.join(args.out, f'{args.task}.json'), 'w') as f:
        json.dump(categories, f)
    plot_bar(categories, args)
    
    
def analysis_area(args):
    if args.label_type == '.json':
        pass
    object_areas = dict()
    object_areas['Tiny'] = 0
    object_areas['Small'] = 0
    object_areas['Medium'] = 0
    object_areas['Large'] = 0
    file_generator = mmengine.scandir(args.label_dir)
    for file in mmengine.track_iter_progress(list(file_generator)):
        with open(os.path.join(args.label_dir, file), 'r') as f:
            lines = f.readlines()
            areas = []
            for line in lines:
                splitline = line.split(' ')
                pts = np.array(list(map(np.float32,splitline[1:9]))).reshape(4, 2)
                (x, y), (w, h), angle = cv2.minAreaRect(pts)
                # bboxes = torch.tensor(list(map(float,splitline[1:9])))
                # rbboxes = poly2obb(bboxes)
                # area = rbboxes[:, 3] * rbboxes[:, 2]
                area = w * h
                if area <= 1000:
                    object_areas['Tiny'] += 1
                elif 1000 < area <=5000:
                    object_areas['Small'] += 1 
                elif 5000 < area <= 10000:
                    object_areas['Medium'] += 1
                elif area > 10000:
                    object_areas['Large'] += 1
    plot_bar(object_areas, args)


def analysis_ratio(args):
    if args.label_type == '.json':
        pass
    widths = []
    heights = []
    file_generator = mmengine.scandir(args.label_dir)
    k=0
    for file in mmengine.track_iter_progress(list(file_generator)):
        with open(os.path.join(args.label_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                splitline = line.split(' ')
                pts = np.array(list(map(np.float32,splitline[1:9]))).reshape(4, 2)
                (x, y), (w, h), angle = cv2.minAreaRect(pts)
                widths.append(w)
                heights.append(h)
    plot_scatter(widths, heights, args)

def main():
    args = parse_args()
    analysis_func = eval(f"analysis_{args.task}")
    analysis_func(args)
    

if __name__ == "__main__":
    main()
