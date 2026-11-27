from model.config import DefaultConfig
import torch
from dataset.voc_dataset_fusion import VOCDataset as VOCDataset
import time
from dataset.augment import Transforms_Fusion
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.fcos_fusion import Network_IJCAI, load_sfdfuse_checkpoint
from architect import Architect as Architect1
from model.loss import Fusionloss_grad2


def get_args():
    parser = argparse.ArgumentParser("TB-BiLFusion Train Script")

    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="training batch size")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n_gpu", type=str, default='0', help="GPU IDs, for example 0 or 0,1")
    parser.add_argument("--data_root", type=str, required=True,
                        help="path to the M3FD dataset root")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_irpreserve",
                        help="checkpoint output directory")
    parser.add_argument("--log_file", type=str, default="loss_galker_q_tr.txt", help="loss log output path")
    parser.add_argument("--save_every", type=int, default=5, help="checkpoint save interval")
    parser.add_argument("--log_flush_every", type=int, default=50, help="log flush interval")
    parser.add_argument("--empty_cache_every", type=int, default=50, help="cuda cache cleanup interval")
    parser.add_argument("--eval_every", type=int, default=5, help="validation mAP interval")
    parser.add_argument("--eval_max_batches", type=int, default=0, help="max validation batches; 0 means all")
    parser.add_argument(
        "--fusion_ckpt",
        type=str,
        default=r"./sfdfuse_stage2_best_change2.pth",
        help="SFDFuse checkpoint used to initialize model.enhance_net.net",
    )


    parser.add_argument('--model_path', type=str, default='saved_models', help='model output directory')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping threshold')
    parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='architecture learning rate')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        help='architecture weight decay')


    parser.add_argument('--implicit_scale', type=float, default=1.0,
                        help='trust coefficient for implicit hypergradient correction')
    parser.add_argument('--hypergrad_clip', type=float, default=5.0,
                        help='hypergradient norm clipping threshold')
    parser.add_argument('--fd_radius', type=float, default=1e-2,
                        help='finite-difference perturbation radius for graph-free HVP')
    parser.add_argument('--robust_topk', type=int, default=3,
                        help='number of worst modality scenarios retaining autograd graphs; 2 balances robustness and memory')
    parser.add_argument('--hypergrad_eps', type=float, default=1e-8,
                        help='numerical epsilon used by stable hypergradient')


    parser.add_argument('--hyper_mode', type=str, default='galker_q_tr',
                        choices=['galker_q_tr'],
                        help='q-probe Galerkin hypergradient with relative trust region')
    parser.add_argument('--task_weight_mode', type=str, default='invnorm',
                        choices=['invnorm'],
                        help='inverse-gradient-norm task weighting')
    parser.add_argument('--outer_mode', type=str, default='dro', choices=['dro'],
                        help='modality-robust smooth-CVaR outer objective')
    parser.add_argument('--cvar_alpha', type=float, default=0.40,
                        help='risk level for modality-robust smooth CVaR outer objective')
    parser.add_argument('--cvar_tau', type=float, default=0.05,
                        help='softplus temperature used to smooth CVaR hinge')
    parser.add_argument('--ir_attenuation', type=float, default=0.70,
                        help='IR modality attenuation scenario')
    parser.add_argument('--vis_attenuation', type=float, default=0.70,
                        help='visible modality attenuation scenario')

    args = parser.parse_args()
    return args


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_order = [(-score).argsort() for score in pred_scores]
    boxes = [boxes[idx] for boxes, idx in zip(pred_boxes, score_order)]
    labels = [labels[idx] for labels, idx in zip(pred_labels, score_order)]
    scores = [scores[idx] for scores, idx in zip(pred_scores, score_order)]
    return boxes, labels, scores


def iou_2d(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    overlap = np.maximum(
        0.0,
        np.minimum(a[..., 2:], b[..., 2:])
        - np.maximum(a[..., :2], b[..., :2]),
    )
    overlap = np.prod(overlap, axis=-1)
    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)
    return overlap / np.maximum(area_a + area_b - overlap, 1e-12)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])
    points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[points + 1] - mrec[points]) * mpre[points + 1]))


def evaluate_map(
    model,
    dataset,
    max_batches=0,
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    gt_boxes, gt_labels = [], []
    pred_boxes, pred_labels, pred_scores = [], [], []
    eval_model = Network_IJCAI(
        None,
        mode="inference",
        config=model.denoise_net.fcos_body.config,
    ).cuda()
    eval_model.load_state_dict(model.state_dict(), strict=False)
    eval_model.eval()
    with torch.no_grad():
        for batch_idx, (ir, vis, _mask, boxes, classes) in enumerate(loader, 1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            _, scores, pred_cls, pred_box = eval_model.forward_object(
                ir.cuda(), vis.cuda(), None, None
            )
            pred_boxes.append(pred_box[0].cpu().numpy())
            pred_labels.append(pred_cls[0].cpu().numpy())
            pred_scores.append(scores[0].cpu().numpy())
            valid = classes[0] >= 0
            gt_boxes.append(boxes[0][valid].numpy())
            gt_labels.append(classes[0][valid].numpy())

    pred_boxes, pred_labels, pred_scores = sort_by_score(
        pred_boxes, pred_labels, pred_scores
    )
    aps = []
    for class_id in range(1, len(dataset.CLASSES_NAME)):
        class_gt = [b[l == class_id] for b, l in zip(gt_boxes, gt_labels)]
        class_pred = [b[l == class_id] for b, l in zip(pred_boxes, pred_labels)]
        class_score = [s[l == class_id] for s, l in zip(pred_scores, pred_labels)]
        tp_all, fp_all, score_all = [], [], []
        total_gt = 0
        for sample_gt, sample_pred, sample_score in zip(
                class_gt, class_pred, class_score
        ):
            total_gt += len(sample_gt)
            assigned = set()
            for pred_idx in np.argsort(-sample_score):
                score = sample_score[pred_idx]
                if len(sample_gt) == 0:
                    tp_all.append(0)
                    fp_all.append(1)
                    score_all.append(score)
                    continue
                overlaps = iou_2d(
                    sample_gt, sample_pred[pred_idx: pred_idx + 1]
                )[:, 0]
                gt_idx = int(np.argmax(overlaps))
                if overlaps[gt_idx] >= 0.5 and gt_idx not in assigned:
                    assigned.add(gt_idx)
                    tp_all.append(1)
                    fp_all.append(0)
                else:
                    tp_all.append(0)
                    fp_all.append(1)
                score_all.append(score)
        if total_gt == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-np.asarray(score_all))
        tp = np.cumsum(np.asarray(tp_all)[order])
        fp = np.cumsum(np.asarray(fp_all)[order])
        recall = tp / total_gt
        precision = tp / np.maximum(tp + fp, 1e-12)
        aps.append(compute_ap(recall, precision))
    del eval_model
    torch.cuda.empty_cache()
    return float(np.mean(aps))


if __name__ == '__main__':
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    transform = Transforms_Fusion()
    config = DefaultConfig()


    train_dataset = VOCDataset(
        root_dir=args.data_root,
        resize_size=[384, 512],
        split='train',
        is_train=True,
        augment=transform
    )
    val_dataset = VOCDataset(
        root_dir=args.data_root,
        resize_size=[384, 512],
        split='val',
        is_train=False,
        augment=None
    )

    fusion_loss = Fusionloss_grad2().cuda()

    model = Network_IJCAI(
        f_loss=fusion_loss,
        mode="training",
        config=config,
    ).cuda()
    fusion_ckpt = args.fusion_ckpt
    if fusion_ckpt:
        if not os.path.exists(fusion_ckpt):
            raise FileNotFoundError(
                f"SFDFuse checkpoint not found: {fusion_ckpt}"
            )
        load_sfdfuse_checkpoint(
            model.enhance_net,
            fusion_ckpt,
            strict=False,
        )
    else:
        print(
            "SFDFuse checkpoint not provided. "
            "The SFDFuse fusion branch will train from scratch."
        )

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    loader_kwargs = dict(
        num_workers=args.n_cpu,
        pin_memory=torch.cuda.is_available(),
    )
    if args.n_cpu > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        **loader_kwargs,
    )

    val_dataset_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        **loader_kwargs,
    )

    val_iter = iter(val_dataset_loader)
    print(f"Total train images: {len(train_dataset)}")
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = int(TOTAL_STEPS * 0.03)
    DECAY_STEP_1 = int(TOTAL_STEPS * 0.6)
    DECAY_STEP_2 = int(TOTAL_STEPS * 0.8)
    GLOBAL_STEPS = 1
    LR_INIT = 2e-4
    DISC_LR_SCALE = 0.5
    FUSION_10STEP_LR_SCALE = 0.5

    architect = Architect1(model, args)


    optimizer_dis = torch.optim.SGD(
        model.discriminator.parameters(),
        lr=LR_INIT,
        momentum=0.9,
        weight_decay=0.0001
    )

    optimizer_detection = torch.optim.SGD(
        model.denoise_net.parameters(),
        lr=LR_INIT,
        momentum=0.9,
        weight_decay=0.0001
    )
    optimizer_fusion = torch.optim.SGD(
        model.enhance_net.parameters(),
        lr=LR_INIT * FUSION_10STEP_LR_SCALE,
        momentum=0.0,
        weight_decay=0.0,
    )

    model.train()
    f_loss_txt = open(args.log_file, 'w', encoding="utf-8")
    f_loss_txt.write(
        "# per-step: det_loss fusion_loss gan_loss lr "
        "(fusion_loss=nan on non-update steps)\n"
    )
    f_loss_txt.write(
        "# arch_stats: task weights, direct/implicit/hyper norms and their shares, "
        "lower/upper losses, lower gradient norms, curvature, FD radius, "
        "cosine, and DRO tail statistics\n"
    )
    best_map = -1.0
    best_epoch = -1

    try:

        for epoch in range(EPOCHS):

            for epoch_step, data in enumerate(train_loader):

                batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes = data


                batch_imgs_ir = batch_imgs_ir.cuda(non_blocking=True)
                batch_imgs_vis = batch_imgs_vis.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                batch_boxes = batch_boxes.cuda(non_blocking=True)
                batch_classes = batch_classes.cuda(non_blocking=True)


                if GLOBAL_STEPS <= WARMPUP_STEPS:

                    current_lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
                elif GLOBAL_STEPS == DECAY_STEP_1:
                    current_lr = LR_INIT * 0.1
                elif GLOBAL_STEPS == DECAY_STEP_2:
                    current_lr = LR_INIT * 0.01
                else:
                    current_lr = (
                        optimizer_dis.param_groups[0]['lr'] / DISC_LR_SCALE
                    )


                for pg in optimizer_dis.param_groups:
                    pg['lr'] = current_lr * DISC_LR_SCALE
                for pg in optimizer_detection.param_groups:
                    pg['lr'] = current_lr
                for pg in optimizer_fusion.param_groups:
                    pg['lr'] = max(
                        current_lr * FUSION_10STEP_LR_SCALE,
                        1e-6,
                    )

                start_time = time.time()
                fusion_loss_val = float("nan")


                if (GLOBAL_STEPS + 1) % 10 == 0:

                    try:
                        val_data = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dataset_loader)
                        val_data = next(val_iter)
                    batch_imgs_ir_old, batch_imgs_vis_old, mask_old, batch_boxes_old, batch_classes_old = val_data

                    batch_imgs_ir_old = batch_imgs_ir_old.cuda(non_blocking=True)
                    batch_imgs_vis_old = batch_imgs_vis_old.cuda(non_blocking=True)
                    mask_old = mask_old.cuda(non_blocking=True)
                    batch_boxes_old = batch_boxes_old.cuda(non_blocking=True)
                    batch_classes_old = batch_classes_old.cuda(non_blocking=True)


                    architect.step(
                        batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes,
                        batch_imgs_ir_old, batch_imgs_vis_old, mask_old, batch_boxes_old, batch_classes_old,
                        lr_new=current_lr
                    )


                    optimizer_fusion.zero_grad(set_to_none=True)
                    set_requires_grad(model.enhance_net, True)
                    loss_fusion = model._fusion_loss_step(
                        batch_imgs_ir, batch_imgs_vis, mask
                    )
                    fusion_loss_val = loss_fusion.mean().item()
                    loss_fusion.mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.enhance_net.parameters(), args.grad_clip
                    )
                    optimizer_fusion.step()
                    del loss_fusion

                    arch_stats = architect.last_stats
                    if arch_stats:
                        v = arch_stats['visual']
                        p = arch_stats['perception']
                        weight_gap = abs(v.weight - p.weight)
                        visual_implicit_ratio = v.implicit_norm / max(v.direct_norm, 1e-12)
                        perception_implicit_ratio = p.implicit_norm / max(p.direct_norm, 1e-12)
                        post_projection_conflict = int(v.raw_cosine_vp < 0)
                        direct_share_sum = v.direct_norm + p.direct_norm + 1e-12
                        implicit_share_sum = v.implicit_norm + p.implicit_norm + 1e-12
                        hyper_share_sum = v.hyper_norm + p.hyper_norm + 1e-12
                        visual_direct_share = v.direct_norm / direct_share_sum
                        perception_direct_share = p.direct_norm / direct_share_sum
                        visual_implicit_share = v.implicit_norm / implicit_share_sum
                        perception_implicit_share = p.implicit_norm / implicit_share_sum
                        visual_hyper_share = v.hyper_norm / hyper_share_sum
                        perception_hyper_share = p.hyper_norm / hyper_share_sum
                        visual_hyper_direct_ratio = v.hyper_norm / max(v.direct_norm, 1e-12)
                        perception_hyper_direct_ratio = p.hyper_norm / max(p.direct_norm, 1e-12)
                        visual_hyper_implicit_ratio = (
                            v.hyper_norm / v.implicit_norm
                            if v.implicit_norm > 1e-12
                            else 0.0
                        )
                        perception_hyper_implicit_ratio = (
                            p.hyper_norm / p.implicit_norm
                            if p.implicit_norm > 1e-12
                            else 0.0
                        )
                        print(
                            f"[ARCH-LOWMEM] mode={args.hyper_mode}/{args.task_weight_mode}/{args.outer_mode} "
                            f"wV={v.weight:.3f} wP={p.weight:.3f} "
                            f"| GV={v.hyper_norm:.4f} GP={p.hyper_norm:.4f} "
                            f"| shares={visual_hyper_share:.3f}/{perception_hyper_share:.3f} "
                            f"| cos={v.cosine_before:.3f}->{v.cosine_after:.3f} "
                            f"| rayV={v.curvature_rayleigh:.3e} rayP={p.curvature_rayleigh:.3e} "
                            f"| skipV={v.solver_skipped} skipP={p.solver_skipped} "
                            f"| cosGQ={v.probe_cosine_gq:.3f}/{p.probe_cosine_gq:.3f} "
                            f"| alpha={v.galerkin_alpha:.3e}/{p.galerkin_alpha:.3e} "
                            f"| alphaRaw={v.galerkin_alpha_raw:.3e}/{p.galerkin_alpha_raw:.3e} "
                            f"| trustD={v.trust_delta:.3e}/{p.trust_delta:.3e} "
                            f"| alphaClip={v.alpha_clipped}/{p.alpha_clipped} "
                            f"| lowerFeas={v.lower_feasible_before:.3e}->{v.lower_feasible_after:.3e} "
                            f"| proj={v.lower_projection_active} "
                            f"| robustSpreadV={v.robust_spread:.4f} robustSpreadP={p.robust_spread:.4f} "
                            f"| tail={v.tail_count}/{p.tail_count}"
                        )
                        f_loss_txt.write(
                            f"global_step:{GLOBAL_STEPS} arch_stats "
                            f"mode:{args.hyper_mode}/{args.task_weight_mode}/{args.outer_mode} "
                            f"wV:{v.weight:.6f} wP:{p.weight:.6f} weight_gap:{weight_gap:.6f} "
                            f"GV_direct:{v.direct_norm:.6f} GV_implicit:{v.implicit_norm:.6f} "
                            f"GV_hyper:{v.hyper_norm:.6f} "
                            f"GV_direct_share:{visual_direct_share:.6f} "
                            f"GV_implicit_share:{visual_implicit_share:.6f} "
                            f"GV_hyper_share:{visual_hyper_share:.6f} "
                            f"GV_hyper_direct_ratio:{visual_hyper_direct_ratio:.6f} "
                            f"GV_hyper_implicit_ratio:{visual_hyper_implicit_ratio:.6f} "
                            f"lower_grad_norm_V:{v.lower_grad_norm:.6f} lower_loss_V:{v.lower_loss:.6f} "
                            f"GP_direct:{p.direct_norm:.6f} GP_implicit:{p.implicit_norm:.6f} "
                            f"GP_hyper:{p.hyper_norm:.6f} "
                            f"GP_direct_share:{perception_direct_share:.6f} "
                            f"GP_implicit_share:{perception_implicit_share:.6f} "
                            f"GP_hyper_share:{perception_hyper_share:.6f} "
                            f"GP_hyper_direct_ratio:{perception_hyper_direct_ratio:.6f} "
                            f"GP_hyper_implicit_ratio:{perception_hyper_implicit_ratio:.6f} "
                            f"lower_grad_norm_P:{p.lower_grad_norm:.6f} lower_loss_P:{p.lower_loss:.6f} "
                            f"implicit_ratio_V:{visual_implicit_ratio:.6f} "
                            f"implicit_ratio_P:{perception_implicit_ratio:.6f} "
                            f"upper_loss_V:{v.upper_loss:.6f} upper_loss_P:{p.upper_loss:.6f} "
                            f"raw_cos_VP:{v.raw_cosine_vp:.6f} "
                            f"cos_before:{v.cosine_before:.6f} cos_after:{v.cosine_after:.6f} "
                            f"post_projection_conflict:{post_projection_conflict} "
                            f"implicit_scale:{getattr(args, 'implicit_scale', 1.0):.6f} "
                            f"hypergrad_clip:{getattr(args, 'hypergrad_clip', 5.0):.6f} "
                            f"curvature_rayleigh_V:{v.curvature_rayleigh:.6e} "
                            f"curvature_rayleigh_P:{p.curvature_rayleigh:.6e} "
                            f"solver_skipped_V:{v.solver_skipped} "
                            f"solver_skipped_P:{p.solver_skipped} "
                            f"probe_cos_gq_V:{v.probe_cosine_gq:.6f} "
                            f"probe_cos_gq_P:{p.probe_cosine_gq:.6f} "
                            f"galerkin_alpha_V:{v.galerkin_alpha:.6e} "
                            f"galerkin_alpha_P:{p.galerkin_alpha:.6e} "
                            f"galerkin_alpha_raw_V:{v.galerkin_alpha_raw:.6e} "
                            f"galerkin_alpha_raw_P:{p.galerkin_alpha_raw:.6e} "
                            f"trust_delta_V:{v.trust_delta:.6e} "
                            f"trust_delta_P:{p.trust_delta:.6e} "
                            f"alpha_clipped_V:{v.alpha_clipped} "
                            f"alpha_clipped_P:{p.alpha_clipped} "
                            f"rayleigh_ratio_V:{v.rayleigh_ratio:.6e} "
                            f"rayleigh_ratio_P:{p.rayleigh_ratio:.6e} "
                            f"lower_feasible_before:{v.lower_feasible_before:.6e} "
                            f"lower_feasible_after:{v.lower_feasible_after:.6e} "
                            f"lower_projection_active:{v.lower_projection_active} "
                            f"fd_radius_V:{v.fd_radius:.6e} fd_radius_P:{p.fd_radius:.6e} "
                            f"robust_spread_V:{v.robust_spread:.6f} "
                            f"robust_spread_P:{p.robust_spread:.6f} "
                            f"tail_count_V:{v.tail_count} tail_count_P:{p.tail_count}\n"
                        )
                        f_loss_txt.flush()
                    del batch_imgs_ir_old, batch_imgs_vis_old, mask_old, batch_boxes_old, batch_classes_old
                    if (GLOBAL_STEPS + 1) % max(args.empty_cache_every, 1) == 0:
                        torch.cuda.empty_cache()


                optimizer_dis.zero_grad(set_to_none=True)
                set_requires_grad(model.enhance_net, False)
                try:
                    loss_disc = model._discriminator_loss(
                        batch_imgs_ir, batch_imgs_vis, mask
                    )
                    loss_disc.mean().backward()
                finally:
                    set_requires_grad(model.enhance_net, True)
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), args.grad_clip)
                optimizer_dis.step()


                optimizer_detection.zero_grad(set_to_none=True)
                set_requires_grad(model.enhance_net, False)
                set_requires_grad(model.discriminator, False)
                try:
                    loss_det = model._detection_loss(batch_imgs_ir, batch_imgs_vis, batch_boxes, batch_classes)
                    loss_det.mean().backward()
                finally:
                    set_requires_grad(model.enhance_net, True)
                    set_requires_grad(model.discriminator, True)
                torch.nn.utils.clip_grad_norm_(model.denoise_net.parameters(), args.grad_clip)
                optimizer_detection.step()

                end_time = time.time()
                cost_time = int((end_time - start_time) * 1000)


                det_loss_val = loss_det.mean().item()
                gan_loss_val = loss_disc.mean().item()


                print(
                    f"global_steps:{GLOBAL_STEPS} epoch:{epoch + 1} "
                    f"steps:{epoch_step + 1}/{steps_per_epoch} "
                    f"cost_time:{cost_time}ms lr={current_lr:.4e} "
                    f"det_loss:{det_loss_val:.4f} "
                    f"fusion_loss:{fusion_loss_val:.4f} "
                    f"gan_loss:{gan_loss_val:.4f}"
                )

                f_loss_txt.write(
                    f"global_step:{GLOBAL_STEPS} det_loss:{det_loss_val:.4f} "
                    f"fusion_loss:{fusion_loss_val:.4f} gan_loss:{gan_loss_val:.4f}\n")
                if (GLOBAL_STEPS + 1) % max(args.log_flush_every, 1) == 0:
                    f_loss_txt.flush()

                GLOBAL_STEPS += 1
                del batch_imgs_ir, batch_imgs_vis, mask, batch_boxes, batch_classes

            if (epoch + 1) % max(args.save_every, 1) == 0:

                save_path = os.path.join(args.checkpoint_dir, f"model_{epoch + 1}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Save checkpoint to {save_path}")

            if (epoch + 1) % max(args.eval_every, 1) == 0:
                val_map = evaluate_map(
                    model,
                    val_dataset,
                    args.eval_max_batches,
                )
                print(f"Validation mAP@0.5 at epoch {epoch + 1}: {val_map:.4f}")
                f_loss_txt.write(f"epoch:{epoch + 1} val_mAP:{val_map:.4f}\n")
                f_loss_txt.flush()
                if val_map > best_map:
                    best_map = val_map
                    best_epoch = epoch + 1
                    best_path = os.path.join(
                        args.checkpoint_dir,
                        f"best_mAP_{args.hyper_mode}_{args.task_weight_mode}.pth",
                    )
                    torch.save(model.state_dict(), best_path)
                    with open(
                            os.path.join(
                                args.checkpoint_dir,
                                f"best_mAP_{args.hyper_mode}_{args.task_weight_mode}.txt",
                            ),
                            "w",
                            encoding="utf-8",
                    ) as best_txt:
                        best_txt.write(
                            f"best_mAP:{best_map:.6f}\nbest_epoch:{best_epoch}\n"
                        )
                    print(f"Save best mAP checkpoint to {best_path}")

    except Exception as e:

        print(f"Training exception: {e}")
        raise
    finally:

        f_loss_txt.close()
        print("Train finished, loss log saved to", args.log_file)
