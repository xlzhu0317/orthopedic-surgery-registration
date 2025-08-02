import argparse
import os.path as osp
import time
import torch

import numpy as np

from src.engine import SingleTester
from src.utils.torch import release_cuda
from src.utils.common import ensure_dir, get_log_string

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['test', 'val'], help='test benchmark')
    return parser


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.register_model(model)

        self.evaluator = Evaluator(cfg).cuda()
        self.output_dir = osp.join(cfg.output_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        torch.cuda.synchronize()
        start_time = time.time()

        output_dict = self.model(data_dict)

        torch.cuda.synchronize()
        end_time = time.time()


        output_dict['inference_time'] = end_time - start_time
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']

        message = f'{scene_name}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        if isinstance(scene_name, list):
            scene_name = scene_name[0]

        file_name = osp.join(self.output_dir, scene_name)

        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            ref_points_f=release_cuda(output_dict['ref_points_f']),
            src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            src_feats_c=release_cuda(output_dict['src_feats_c']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],

            inference_time=output_dict.get('inference_time', -1.0),
            PIR=result_dict.get('PIR', torch.tensor(-1.0)).item(),
            IR=result_dict.get('IR', torch.tensor(-1.0)).item()
        )
        # *************************************************


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()