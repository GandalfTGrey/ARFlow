import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from utils.flow_utils import load_flow, evaluate_flow
from utils.misc_utils import AverageMeter
from utils.mono_utils import stitching_and_show
import torchvision

import matplotlib.pyplot as plt
import matplotlib.colors as colors
cmap = plt.get_cmap('viridis')
import utils.mono_utils as mono_utils
import os
from datetime import datetime


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            res_dict = self.model(img_pair, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]
            loss, l_ph, l_sm, flow_mean, occ_mask = self.loss_func(flows, img_pair)

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
                              img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            # loss.backward()

            scaled_loss = 1024. * loss
            scaled_loss.backward()

            for param in [p for p in self.model.parameters() if p.requires_grad]:
                param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)
                
                for j in range(min(4, img1.size()[0])):
                    # add flow visulization:
                    to_tb_vis = stitching_and_show([img1[j], img2[j], flows_12[0][j], flows_21[0][j], occ_mask[j]], ver=True, show=False)
                    to_tb_vis = torchvision.transforms.functional.pil_to_tensor(to_tb_vis)
                    self.summary_writer.add_image('img1, img2, flow12, flow21, occ_mask/{}_th_batch'.format(j), to_tb_vis, self.i_iter)


                for i in range():
                    to_tb_vis = stitching_and_show([flows_12[i][0], flows_21[i][0], occ_mask[i][0]], ver=True, show=False)
                    to_tb_vis = torchvision.transforms.functional.pil_to_tensor(to_tb_vis)
                    self.summary_writer.add_image('MultiScale,flow12, flow21, occ_mask/scale{}'.format(i), to_tb_vis, self.i_iter)

                    
            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info:loss,l_ph,l_sm,flow_meam{}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        import cv2
        batch_time = AverageMeter()

        if type(self.valid_loader) is not list:
            self.valid_loader = [self.valid_loader]

        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        self.model = self.model.module
        self.model.eval()
        end = time.time()

        all_error_names = []
        all_error_avgs = []
        save_path_dir = os.path.join(self.save_root, "kitti_sceneflow_eval")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
            
        def vis_flow_eval(flows, gt_flows, device=self.device, save_path_dir=save_path_dir):
            print(save_path_dir)
            i = 0 # batch idx
            flow_out = flows[0][i]  # size[2, 256, 832]
            flow_gt = torch.tensor(gt_flows[i][:,:,0:2], device=self.device).permute((2,0,1))
            valid_occ = torch.tensor(gt_flows[i][:,:,2], device=self.device)
            valid_noc = torch.tensor(gt_flows[i][:,:,3], device=self.device)
            _, h, w = flow_out.shape
            _, H, W = flow_gt.shape
            flow_out[0, :, :] = flow_out[0, :, :] / w * W
            flow_out[1, :, :] = flow_out[1, :, :] / h * H
            trans = torchvision.transforms.Resize((H, W), antialias=True)
            flow_out = trans(flow_out)
            
            
            err_map = torch.sum(torch.abs(flow_out - flow_gt) * valid_occ, dim=0).cpu()
            err_map_norm = colors.Normalize(vmin=0, vmax=torch.max(err_map))
            err_map_colored_tensor = mono_utils.plt_color_map_to_tensor(cmap(err_map_norm(err_map)))
            to_save = mono_utils.stitching_and_show(img_list=[flow_out, flow_gt, err_map_colored_tensor, img1[i], img2[i]],
                                                    ver=True, show=False)
            save_path = os.path.join(save_path_dir, str(self.i_epoch) + "th_epoch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".png")
            to_save.save(save_path)
            
        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['EPE', 'E_noc', 'E_occ', 'F1_all']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)

                res = list(map(load_flow, data['flow_occ']))
                gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
                res = list(map(load_flow, data['flow_noc']))
                _, noc_masks = [r[0] for r in res], [r[1] for r in res]

                gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                            flow, occ_mask, noc_mask in
                            zip(gt_flows, occ_masks, noc_masks)]

                # compute output
                flows = self.model(img_pair)['flows_fw']
                pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
                
                ###########################################
                ############## visualization ##############
                ###########################################

                # if self.i_epoch % 10 == 0:
                if True:
                    vis_flow_eval(flows, gt_flows, self.device)

                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img_pair.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar(
                    'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0], name='KITTI_Flow')

        return all_error_avgs, all_error_names
