from .head import ClsCntRegHead
from .fpn_neck import FPN
from .backbone.resnet import resnet50
import torch.nn as nn
from .loss import GenTargets, LOSS, coords_fmap2orig, Fusionloss_grad2
import torch
from .config import DefaultConfig
from .sfdfuse_net import UnifiedFusionNetwork
import torch.nn.functional as F
import functools


class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrained=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False
        if mode and getattr(self.config, "freeze_bn", False):
            self.apply(freeze_bn)


    def forward(self, x):
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))
        cls_classes = cls_classes + 1

        boxes = self._coords2boxes(coords, reg_preds)


        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])
            _boxes.append(boxes[batch][topk_ind[batch]])
        cls_scores_topk = torch.stack(_cls_scores, dim=0)
        cls_classes_topk = torch.stack(_cls_classes, dim=0)
        boxes_topk = torch.stack(_boxes, dim=0)
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        if cls_classes_topk.shape[0] != 1:
            raise ValueError(
                "DetectHead currently supports batch_size=1 in inference."
            )
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]
            _cls_classes_b = cls_classes_topk[batch][mask]
            _boxes_b = boxes_topk[batch][mask]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.as_tensor(
            keep,
            dtype=torch.long,
            device=boxes.device,
        )

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)


        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]
        boxes = torch.cat([x1y1, x2y2], dim=-1)
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", config=None):
        super().__init__()

        if config is None:
            config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config)

        if mode == "training":


            self.target_layer = GenTargets(strides=config.strides, limit_range=config.limit_range)

            self.loss_layer = LOSS()

        elif mode == "inference":


            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)

            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == "inference":

            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]],
        device=input_im.device,
        dtype=input_im.dtype,
    )
    bias = torch.tensor(
        [0.0 / 255, -0.5, -0.5],
        device=input_im.device,
        dtype=input_im.dtype,
    )
    temp = (im_flat + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                ndf,
                ndf * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
            ),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                ndf * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
            ),
        )

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class SFDFuseGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = UnifiedFusionNetwork()

    def forward(self, ir, vis):
        ir_y = ir[:, 0:1, :, :]
        vis_y = vis[:, 0:1, :, :]
        fused, _ = self.net(vis_y, ir_y)
        return fused


def load_sfdfuse_checkpoint(enhance_net, checkpoint_path, strict=True):
    if not isinstance(enhance_net, SFDFuseGenerator):
        raise TypeError("enhance_net must be an SFDFuseGenerator")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("unified_network", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported SFDFuse checkpoint type: {type(checkpoint).__name__}"
        )

    expected_keys = set(enhance_net.net.state_dict().keys())
    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        if normalized_key.startswith("enhance_net.net."):
            normalized_key = normalized_key[len("enhance_net.net.") :]
        elif normalized_key not in expected_keys:
            continue
        state_dict[normalized_key] = value

    if not state_dict:
        raise ValueError("No tensor parameters found in the SFDFuse checkpoint")

    result = enhance_net.net.load_state_dict(state_dict, strict=strict)
    print(
        "Loaded SFDFuse fusion checkpoint: "
        f"{checkpoint_path} | missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}"
    )
    return result


def per_image_minmax(image, eps=1e-6):
    batch = image.shape[0]
    flattened = image.reshape(batch, -1)
    min_value = flattened.min(dim=1, keepdim=True).values
    max_value = flattened.max(dim=1, keepdim=True).values
    normalized = (flattened - min_value) / (max_value - min_value + eps)
    return normalized.reshape_as(image)


class Network_IJCAI(nn.Module):

    def __init__(self, f_loss=None, mode="training", config=None):
        super(Network_IJCAI, self).__init__()
        self._criterion = f_loss if f_loss is not None else Fusionloss_grad2()
        self._criterion_gan = GANLoss()

        self.enhance_net = SFDFuseGenerator()
        self.denoise_net = FCOSDetector(mode, config)
        self.discriminator = PixelDiscriminator(1)
        self.mode = mode


    def forward(self, ir, vis, batch_boxes, batch_classes):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)


        fused_rgb = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_rgb = YCrCb2RGB(fused_rgb)
        ones = torch.ones_like(fused_rgb)
        zeros = torch.zeros_like(fused_rgb)
        fused_rgb = torch.where(fused_rgb > ones, ones, fused_rgb)
        fused_rgb = torch.where(fused_rgb < zeros, zeros, fused_rgb)


        fused_rgb = per_image_minmax(fused_rgb)

        if self.mode == "training":
            losses = self.denoise_net(
                [fused_rgb, batch_boxes, batch_classes]
            )
            return fused, losses
        else:
            scores, classes, boxes = self.denoise_net(fused_rgb)
            return fused_rgb, scores, classes, boxes


    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused


    forward_object = forward

    def _fusion_loss_step(self, ir, vis, mask):
        fused_img = self.forward_fusion(ir, vis)
        return self._criterion(ir, vis, mask, fused_img)


    def _discriminator_loss(self, ir, vis, mask):
        fused_img = self.forward_fusion(ir, vis)

        real_img = mask[:, 0:1, :, :]

        loss_fake = self._criterion_gan(
            self.discriminator(fused_img), False
        )
        loss_real = self._criterion_gan(
            self.discriminator(real_img), True
        )
        return (loss_fake + loss_real) * 0.5


    def _generator_loss(self, ir, vis, mask):
        fused_img = self.forward_fusion(ir, vis)

        vis_ycbcr = RGB2YCrCb(vis)
        ir_y = ir[:, 0:1, :, :]
        vis_y = vis_ycbcr[:, 0:1, :, :]

        loss_adv = self._criterion_gan(
            self.discriminator(fused_img), True
        )

        def _grad_map(x):
            gx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            gy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            gx = F.pad(gx, (0, 0, 0, 1))
            gy = F.pad(gy, (0, 1, 0, 0))
            return gx + gy

        grad_vis = _grad_map(vis_y)
        grad_ir = _grad_map(ir_y)
        grad_target = torch.maximum(
            grad_vis,
            0.5 * grad_ir,
        )
        grad_loss = F.l1_loss(
            _grad_map(fused_img),
            grad_target,
        )

        return loss_adv + 1.0 * grad_loss


    def _detection_loss(self, ir, vis, batch_boxes, batch_classes):
        fused_img, losses = self.forward_object(ir, vis, batch_boxes, batch_classes)
        loss_D_fake = self._criterion_gan(
            self.discriminator(fused_img), True
        )
        denoise_loss = losses[-1]
        return denoise_loss + loss_D_fake * 0.01
