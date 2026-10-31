
import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from torch import nn
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
import random


import math
import numpy as np
import copy
from utils.graphics_utils import getWorld2View2

def pcc_loss(pred_depth, gt_depth, mask=None):
    if mask is not None:
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    
    if pred_depth.numel() < 10:
        return torch.tensor(0.0, device=pred_depth.device)

    pred_mean = pred_depth.mean()
    gt_mean = gt_depth.mean()
    pred_centered = pred_depth - pred_mean
    gt_centered = gt_depth - gt_mean

    cov = (pred_centered * gt_centered).sum()
    pred_var = (pred_centered ** 2).sum()
    gt_var = (gt_centered ** 2).sum()

    pcc = cov / torch.sqrt(pred_var * gt_var + 1e-8)
    return 1.0 - pcc

def generate_pseudo_view(viewpoint_cam, perturb_angle_deg=5.0, perturb_trans=0.02):
    from scipy.spatial.transform import Rotation as R
    pseudo_cam = copy.deepcopy(viewpoint_cam)
    
    angle_rad = math.radians(random.uniform(-perturb_angle_deg, perturb_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    rot_tensor = torch.tensor(R.from_rotvec(axis * angle_rad).as_matrix(), dtype=torch.float32, device="cuda")
    
    trans_tensor = (torch.rand(3, device="cuda") - 0.5) * 2 * perturb_trans
    
    old_R = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device="cuda")
    new_R = old_R @ rot_tensor
    old_cam_center = viewpoint_cam.camera_center.cuda()
    new_cam_center = old_cam_center + trans_tensor
    
    new_T = -torch.matmul(new_R.transpose(0, 1), new_cam_center)
    pseudo_cam.R = new_R.cpu().numpy()
    pseudo_cam.T = new_T.cpu().numpy()
    pseudo_cam.camera_center = new_cam_center.clone()
    pseudo_cam.world_view_transform = torch.tensor(getWorld2View2(pseudo_cam.R, pseudo_cam.T, np.zeros(3), 1.0)).transpose(0, 1).cuda()
    pseudo_cam.full_proj_transform = (pseudo_cam.world_view_transform.unsqueeze(0).bmm(pseudo_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    
    return pseudo_cam

def render_depth_hack(viewpoint_cam, gaussians, pipe):
    xyz_homo = torch.cat([gaussians.get_xyz, torch.ones_like(gaussians.get_xyz[:, :1])], dim=-1)
    xyz_cam = torch.matmul(xyz_homo, viewpoint_cam.world_view_transform)
    z_cam = xyz_cam[:, 2:3]
    
    depth_color = z_cam.repeat(1, 3) 
    bg_depth = torch.full((3,), 10.0, device="cuda")
    
    render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg_depth, mult=1.0, override_color=depth_color, is_train=False, iteration=0)
    return render_pkg["render"][0:1, :, :]

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    test_imgs_dir = os.path.join(args.model_path, "test_imgs/")
    os.makedirs(test_imgs_dir, exist_ok = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg_mask = None
    loss_accum = 0
    pseudo_stack = None
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if os.environ.get("GAUSS_CSV") and iteration % 200 == 0:
            with open(os.environ["GAUSS_CSV"], "a") as _fh:
                _fh.write(f"{iteration},{gaussians.get_xyz.shape[0]}\n")

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = getattr(viewpoint_cam, 'gt_alpha_mask', None)
        
        is_pseudo_view = False
        if iteration > 500 and random.random() < args.pseudo_prob: 
            is_pseudo_view = True
            render_cam = generate_pseudo_view(viewpoint_cam, perturb_angle_deg=8.0, perturb_trans=0.03)
        else:
            render_cam = viewpoint_cam

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        if not is_pseudo_view and gt_mask is not None:
             gt_image = gt_image * gt_mask + bg[:, None, None] * (1.0 - gt_mask)

        render_pkg = render_fastgs(render_cam, gaussians, pipe, bg, opt.mult, is_train=(args.is_drop and not is_pseudo_view), iteration=iteration, drop_rate=args.drop_rate)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss = 0.0
        
        if not is_pseudo_view:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = loss + Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            
        if hasattr(render_cam, 'voxel_depth') and render_cam.voxel_depth is not None:
            voxel_depth = render_cam.voxel_depth.cuda()
            rendered_depth = render_depth_hack(render_cam, gaussians, pipe)
            mask = (voxel_depth > 0.1) & (voxel_depth < 9.0)
            
            loss_pcc = pcc_loss(rendered_depth, voxel_depth, mask=mask.squeeze())
            loss = loss + args.lambda_pcc * loss_pcc

            background_mask = ~mask
            if background_mask.any():
                floaters = rendered_depth[background_mask.squeeze(0)]
                floater_penalty = torch.relu(2.0 - floaters).mean()
                loss = loss + 1.0 * floater_penalty
        
        if type(loss) is float:
            loss = image.sum() * 0.0
        if args.opacity_reg > 0.0:
            loss = loss + args.opacity_reg * gaussians.get_opacity.mean()
        if args.scale_reg > 0.0:
            loss = loss + args.scale_reg * gaussians.get_scaling.max(dim=1).values.mean()

            
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            if iteration > opt.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            optim_start.record()
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    if args.variant in ('absgs', 'taming'):
                        importance_score = pruning_score = None
                    else:
                        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                        if args.no_vcp:
                            pruning_score = None

                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = args.prune_min_opacity, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            if iteration % 8000 == 0 and args.variant == 'fastgs' and not args.no_vcp:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                    
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration % 16 == 0:
                    gaussians.shoptimizer.step()
                    gaussians.shoptimizer.zero_grad(set_to_none = True)
            
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    if args.final_prune_at_end:
        camlist = sampling_cameras(scene.getTrainCameras().copy())
        _, pruning_score_end = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
        _before = gaussians._xyz.shape[0]
        gaussians.final_prune_fastgs(min_opacity=args.final_prune_min_opacity,
                                     pruning_score=pruning_score_end,
                                     score_thresh=args.final_prune_score_thresh)
        print(f"[OURS final_prune] {_before} -> {gaussians._xyz.shape[0]}")

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000,])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument('--lambda_pcc', type=float, default=0.5, help='Weight for PCC depth loss')
    parser.add_argument('--pseudo_prob', type=float, default=0.3, help='Probability of generating a pseudo-view per iteration')
    parser.add_argument('--is_drop', action='store_true', default=False, help='Whether to use DropGaussian logic (default False)')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Drop rate for FastGS (0.0 means completely off)')
    parser.add_argument('--prune_min_opacity', type=float, default=0.001,
                        help='opacity threshold of densify_and_prune_fastgs (upstream hardcoded 0.001)')
    parser.add_argument('--final_prune_at_end', action='store_true', default=False,
                        help='run final_prune_fastgs once more after the last iteration')
    parser.add_argument('--variant', type=str, default='fastgs', choices=['fastgs','absgs','taming'],
                        help='fastgs = VCD+VCP; absgs = abs-grad only; taming = vanilla 3DGS-accel base')
    parser.add_argument('--absgrad_clone', action='store_true', default=False, help='use abs gradient for cloning (AbsGS)')
    parser.add_argument('--no_vcp', action='store_true', default=False,
                        help='ablation: keep VCD densification, disable view-consistency pruning')
    parser.add_argument('--final_prune_min_opacity', type=float, default=0.1)
    parser.add_argument('--final_prune_score_thresh', type=float, default=0.9)
    parser.add_argument('--opacity_reg', type=float, default=0.0,
                        help='L1 weight on per-Gaussian opacity; pushes haze/floaters to zero')
    parser.add_argument('--scale_reg', type=float, default=0.0,
                        help='weight on the largest per-Gaussian scale; pushes oversized blobs down')
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
