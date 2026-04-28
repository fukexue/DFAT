from typing import Dict

import torch
import ipdb
from tqdm import tqdm

from geotransformer.engine.base_tester import BaseTester
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string
from geotransformer.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, args, cfg, cudnn_deterministic=True):
        super().__init__(args, cfg, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            # if True:
            #     from geotransformer.utils.pointcloud import apply_transform_tensor
            #     from copy import deepcopy
            #     data_dict_cur = deepcopy(data_dict)
            #     for idx in range(len(data_dict_cur['points'])):
            #         s2r = apply_transform_tensor(output_dict['estimated_transform'], data_dict_cur['points'][idx][data_dict_cur['lengths'][idx][0]:])
            #         s2randr = torch.cat([data_dict_cur['points'][idx][:data_dict_cur['lengths'][idx][0]], s2r], dim=0)
            #         data_dict_cur['points'][idx] = s2randr
            #     output_dict_cur = self.test_step(self.iteration, data_dict_cur)
            #     output_dict['estimated_transform'] = torch.matmul(output_dict_cur['estimated_transform'], output_dict['estimated_transform'])
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # if result_dict['RR']<1:
            #     from geotransformer.utils.visualization_v1 import viz_flow_mayavi, viz_coarse_nn_correspondence_mayavi
            #     from geotransformer.utils.pointcloud import apply_transform_tensor
            #     # src_init = output_dict['src_points_f'].cpu()
            #     # # s_pc_deformed=None
            #     # s_pc_deformed = apply_transform_tensor(output_dict['estimated_transform'].cpu(), output_dict['src_points_f'].cpu())
            #     # src_gt = apply_transform_tensor(data_dict['transform'].cpu(), output_dict['src_points_f'].cpu())
            #     # viz_flow_mayavi(s_pc=src_init, s_pc_deformed=s_pc_deformed, s_pc_gt=src_gt, t_pc=output_dict['ref_points_f'].cpu())
            #     correspondence_corase = torch.cat([output_dict['src_node_corr_indices'].unsqueeze(0).cpu(),
            #                                        output_dict['ref_node_corr_indices'].unsqueeze(0).cpu()], dim=0)
            #     # gt transformer
            #     f_src_pcd_gt = apply_transform_tensor(data_dict['transform'].cpu(), output_dict['src_points_f'].cpu())
            #     c_src_pcd_gt = apply_transform_tensor(data_dict['transform'].cpu(), output_dict['src_points_c'].cpu())
            #     viz_coarse_nn_correspondence_mayavi(s_pc=c_src_pcd_gt, t_pc=output_dict['ref_points_c'].cpu(),
            #                                         correspondence=correspondence_corase,
            #                                         f_src_pcd=f_src_pcd_gt, f_tgt_pcd=output_dict['ref_points_f'].cpu())
            #     # est transformer
            #     f_src_pcd_gt = apply_transform_tensor(output_dict['estimated_transform'].cpu(), output_dict['src_points_f'].cpu())
            #     c_src_pcd_gt = apply_transform_tensor(output_dict['estimated_transform'].cpu(), output_dict['src_points_c'].cpu())
            #     viz_coarse_nn_correspondence_mayavi(s_pc=c_src_pcd_gt, t_pc=output_dict['ref_points_c'].cpu(),
            #                                         correspondence=correspondence_corase,
            #                                         f_src_pcd=f_src_pcd_gt, f_tgt_pcd=output_dict['ref_points_f'].cpu())
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
            torch.cuda.empty_cache()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
