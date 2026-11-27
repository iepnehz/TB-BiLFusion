import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import DefaultConfig


def coords_fmap2orig(feature, stride):
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        cls_logits = cls_logits.permute(0, 2, 3, 1)
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])

        off_min = torch.min(ltrb_off, dim=-1)[0]
        off_max = torch.max(ltrb_off, dim=-1)[0]

        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu

        mask_pos = mask_in_gtboxes & mask_in_level & mask_center

        areas[~mask_pos] = 99999999
        areas_min_ind = torch.min(areas, dim=-1)[1]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
                                                                                  1)]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]
        cls_targets = classes[
            torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)


        mask_pos_2 = mask_pos.long().sum(dim=-1)


        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        cls_targets[~mask_pos_2] = 0
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        return cls_targets, cnt_targets, reg_targets


def compute_cls_loss(preds, targets, mask):
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)

    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]
        target_pos = targets[batch_index]
        target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,
                      :] == target_pos).float()
        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0) / num_pos


def compute_cnt_loss(preds, targets, mask):
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)

    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]
        target_pos = targets[batch_index][mask[batch_index]]
        assert len(pred_pos.shape) == 1
        loss.append(
            nn.functional.binary_cross_entropy_with_logits(input=pred_pos, target=target_pos, reduction='sum').view(1))
    return torch.cat(loss, dim=0) / num_pos


def compute_reg_loss(preds, targets, mask, mode='giou'):
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []

    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]
        target_pos = targets[batch_index][mask[batch_index]]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            loss.append(iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            loss.append(giou_loss(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss, dim=0) / num_pos


def iou_loss(preds, targets):
    lt = torch.min(preds[:, :2], targets[:, :2])
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (rb + lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    iou = overlap / (area1 + area2 - overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss.sum()


def giou_loss(preds, targets):
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min + lt_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]

    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    return loss.sum()


def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    pt = pt.clamp_min(1e-6)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()


class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()
        if self.config.add_centerness:
            total_loss = cls_loss + cnt_loss + reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss
        else:
            total_loss = cls_loss + reg_loss + cnt_loss * 0.0
            return cls_loss, cnt_loss, reg_loss, total_loss


from pytorch_ssim import ssim


class Fusionloss_grad2(nn.Module):
    def __init__(self):
        super(Fusionloss_grad2, self).__init__()
        self.target_l1_weight = 0.15
        self.target_gradient_weight = 0.10
        self.target_ssim_weight = 0.10
        self.source_intensity_weight = 0.50

        self.source_gradient_weight = 0.90
        self.source_ssim_weight = 0.50
        self.frequency_weight = 0.05
        self.phase_weight = 0.15
        self.cc_weight = 0.40
        self.scd_weight = 0.30
        self.highfreq_weight = 0.08

    @staticmethod
    def _luminance(image):
        if image.shape[1] == 1:
            return image
        return (
            0.299 * image[:, 0:1, :, :]
            + 0.587 * image[:, 1:2, :, :]
            + 0.114 * image[:, 2:3, :, :]
        ).clamp(0.0, 1.0)

    @staticmethod
    def _pearson_correlation(x, y, eps=1e-6):
        x = x.flatten(1)
        y = y.flatten(1)
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        numerator = (x * y).sum(dim=1)
        denominator = torch.sqrt(
            (x * x).sum(dim=1) * (y * y).sum(dim=1) + eps
        )
        return numerator / denominator

    @staticmethod
    def _sobel_gradient(image):
        sobel_x = image.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = image.new_tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        gx = F.conv2d(image, sobel_x, padding=1)
        gy = F.conv2d(image, sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def _multiscale_gradient_loss(self, fused, ir, vis):
        total = fused.new_zeros(())
        norm = 0.0
        for scale, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
            if scale == 1:
                fused_s, ir_s, vis_s = fused, ir, vis
            else:
                fused_s = F.avg_pool2d(fused, kernel_size=scale, stride=scale)
                ir_s = F.avg_pool2d(ir, kernel_size=scale, stride=scale)
                vis_s = F.avg_pool2d(vis, kernel_size=scale, stride=scale)
            target = torch.maximum(
                self._sobel_gradient(ir_s),
                self._sobel_gradient(vis_s),
            )
            total = total + weight * F.l1_loss(
                self._sobel_gradient(fused_s), target
            )
            norm += weight
        return total / max(norm, 1e-6)

    @staticmethod
    def _frequency_magnitude(image):
        return torch.abs(torch.fft.rfft2(image)) + 1e-7

    @staticmethod
    def _phase(image):
        return torch.angle(torch.fft.rfft2(image))

    def _high_frequency(self, image):
        local_mean = F.avg_pool2d(
            image,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        return image - local_mean

    def forward(self, image_ir, image_vis, mask, generate_img):
        ir = self._luminance(image_ir).clamp(0.0, 1.0)
        vis = self._luminance(image_vis).clamp(0.0, 1.0)
        target = self._luminance(mask).clamp(0.0, 1.0)
        fused = self._luminance(generate_img).clamp(0.0, 1.0)

        target_l1 = F.l1_loss(fused, target)
        target_gradient = F.l1_loss(
            self._sobel_gradient(fused),
            self._sobel_gradient(target),
        )
        target_ssim = 1.0 - ssim(fused, target)
        target_loss = (
            self.target_l1_weight * target_l1
            + self.target_gradient_weight * target_gradient
            + self.target_ssim_weight * target_ssim
        )

        source_intensity = F.l1_loss(fused, torch.maximum(ir, vis))
        source_gradient = self._multiscale_gradient_loss(fused, ir, vis)
        source_ssim = 1.0 - (
            0.8 * ssim(fused, vis) + 0.2 * ssim(fused, ir)
        )

        ir_frequency = self._frequency_magnitude(ir)
        vis_frequency = self._frequency_magnitude(vis)
        frequency_loss = F.l1_loss(
            self._frequency_magnitude(fused),
            torch.maximum(ir_frequency, vis_frequency),
        )

        ir_phase = self._phase(ir)
        vis_phase = self._phase(vis)
        phase_target = torch.atan2(
            torch.sin(ir_phase) + torch.sin(vis_phase),
            torch.cos(ir_phase) + torch.cos(vis_phase),
        )
        phase_loss = torch.mean(
            1.0 - torch.cos(self._phase(fused) - phase_target)
        )

        cc_loss = 1.0 - 0.5 * (
            self._pearson_correlation(fused, ir)
            + self._pearson_correlation(fused, vis)
        ).mean()
        scd_loss = 1.0 - 0.5 * (
            self._pearson_correlation(fused - vis, ir)
            + self._pearson_correlation(fused - ir, vis)
        ).mean()

        fused_hf = self._high_frequency(fused)
        ir_hf = self._high_frequency(ir)
        vis_hf = self._high_frequency(vis)
        highfreq_target = torch.where(
            ir_hf.abs() >= vis_hf.abs(),
            ir_hf,
            vis_hf,
        )
        highfreq_loss = F.l1_loss(fused_hf, highfreq_target)

        return (
            target_loss
            + self.source_intensity_weight * source_intensity
            + self.source_gradient_weight * source_gradient
            + self.source_ssim_weight * source_ssim
            + self.frequency_weight * frequency_loss
            + self.phase_weight * phase_loss
            + self.cc_weight * cc_loss
            + self.scd_weight * scd_loss
            + self.highfreq_weight * highfreq_loss
        )


