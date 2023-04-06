import os
import os.path as osp
import re
import torch
import numpy as np
from typing import Sequence
import zipfile
from mmrotate.structures.bbox import rbox2qbox, qbox2rbox
from mmcv.ops import nms_quadri, nms_rotated
from collections import defaultdict
import glob
from tqdm import tqdm
import argparse


METAINFO = {
        # 'classes': ('C', 'F', 'B', 'K', 'J', 'I', 'H', 'D', 'E', 'G', 'A','xujin'),
        'classes': ('C', 'F', 'B', 'K', 'J', 'I', 'H', 'D', 'E', 'G', 'A'),
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis data')
    parser.add_argument("--src_path", default='', type=str)
    parser.add_argument("--dst_path", type=str, default='')

    args = parser.parse_args()
    return args

def merge_results_tianzhi(src_dir, outfile_prefix, iou_thr=0.01) -> str:

    collector = defaultdict(list)

    if not os.path.exists(outfile_prefix):
        os.mkdir(outfile_prefix)
    results = glob.glob(os.path.join(src_dir, '*.txt'))
    results_list = []
    # result_dict = dict()
    for result in tqdm(results):
        img_name = result.split('/')[-1][:-4]
        # if not img_name in results_dict:
        result_dict = dict()
        result_dict['img_id'] = img_name
            
        with open(result, 'r') as f:
            lines = f.readlines()
            labels = []
            bboxes = []
            scores = []
            for line in lines:
                splitline = line.split(' ')
                labels.append(splitline[-2])
                scores.append(splitline[-1])
                bboxes.append(splitline[:8])
            np_bboxs = np.array(bboxes, dtype=np.float32)
            if len(np_bboxs) == 0:
                np_bboxs = np.zeros((0,8))
            # else:
            #     np_bboxs = np.asarray(qbox2rbox(torch.from_numpy(np_bboxs)))
            #     np_bboxs = np.asarray(qbox2rbox(torch.from_numpy(np_bboxs)))
            np_labels = np.array(labels, dtype=np.int64)
            np_scores = np.array(scores, dtype=np.float32)
            result_dict['scores'] = np_scores
            result_dict['labels'] = np_labels
            result_dict['bboxes'] = np_bboxs
            assert len(np_bboxs) == len(np_labels) == len(np_scores)
        results_list.append(result_dict)
        
    for idx, result in enumerate(results_list):
        img_id = result.get('img_id', idx)
        splitname = img_id.split('__')
        oriname = splitname[0]
        pattern1 = re.compile(r'__\d+___\d+')
        x_y = re.findall(pattern1, img_id)
        x_y_2 = re.findall(r'\d+', x_y[0])
        x, y = int(x_y_2[0]), int(x_y_2[1])
        labels = result['labels']
        bboxes = result['bboxes']
        scores = result['scores']
        ori_bboxes = bboxes.copy()
        # if self.predict_box_type == 'rbox':
        #     ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
        #         [x, y], dtype=np.float32)
        # elif self.predict_box_type == 'qbox':
        ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
            [x, y, x, y, x, y, x, y], dtype=np.float32)
        # else:
        #     raise NotImplementedError
        label_dets = np.concatenate(
            [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
            axis=1)
        collector[oriname].append(label_dets)

    id_list, dets_list = [], []
    for oriname, label_dets_list in collector.items():
        # big_img_results = []
        label_dets = np.concatenate(label_dets_list, axis=0)
        if len(label_dets) == 0:
            continue
        labels, dets = label_dets[:, 0], label_dets[:, 1:]
        try:
            cls_dets = torch.from_numpy(dets).cuda()
            labels = torch.from_numpy(labels).cuda()
        except:  # noqa: E722
            cls_dets = torch.from_numpy(dets)
            labels = torch.from_numpy(labels)
        
        # if self.predict_box_type == 'rbox':
        #     nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], 
        #                                         cls_dets[:, -1], 
        #                                         self.iou_thr)
        # elif self.predict_box_type == 'qbox':
        nms_dets, keep_inds = nms_quadri(cls_dets[:, :8],
                                            cls_dets[:, -1], 
                                            iou_thr)
        # else:
        #     raise NotImplementedError
        nms_labels = labels[keep_inds]
        nms_labels_dets = torch.cat((nms_labels[:, None], nms_dets), dim=-1)

        nms_labels_dets = nms_labels_dets.cpu().numpy()
        # big_img_results.append(nms_labels_dets)
        id_list.append(oriname)
        dets_list.append(nms_labels_dets)            

    # if osp.exists(outfile_prefix):
    #     raise ValueError(f'The outfile_prefix should be a non-exist path, '
    #                         f'but {outfile_prefix} is existing. '
    #                         f'Please delete it firstly.')
    # os.makedirs(outfile_prefix)

    files = [
        osp.join(outfile_prefix, img + '.txt')
        for img in id_list
    ]
    # for file in tqdm(files):
    
    for index, (img_id, dets) in tqdm(enumerate(zip(id_list, dets_list))):
        file = files[index]
        with open(file, 'a') as f:

            if len(dets) == 0:
                continue
            # th_dets = torch.from_numpy(dets)
            str_format = '{} {:.2f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'
            for det in dets:
                outs = str_format.format(METAINFO['classes'][int(det[0])], det[9], det[1], det[2], det[3], 
                                            det[4], det[5], det[6], det[7], det[8])
                f.write(outs)


            # f.close()
                # qboxes = str(dets[2:])

                # qboxes, scores = torch.split(dets, (8, 1), dim=-1)
            
                # for idx, (qbox, score) in enumerate(zip(qboxes, scores)):
                #     classname = METAINFO['classes'][int(dets[idx, 0].item())]
                #     txt_element = [classname, str(round(float(score), 2))
                #                     ] + [f'{p:.2f}' for p in qbox]
                #     f.writelines(' '.join(txt_element) + '\n')

   

    # target_name = osp.split(outfile_prefix)[-1]
    # zip_path = osp.join(outfile_prefix, target_name + '.zip')
    # with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
    #     for f in files:
    #         t.write(f, osp.split(f)[-1])

    # return zip_path

def main():
    args = parse_args()

    merge_results_tianzhi(args.src_path, args.dst_path)


if __name__ == "__main__":
    main()
