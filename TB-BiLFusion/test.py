import time
import torch
from torchvision import transforms
from model.fcos_fusion import Network_IJCAI as Network
import os
import argparse
import numpy as np
import cv2

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)

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


def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)

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


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _index_images(directory):
    images = {}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        stem, extension = os.path.splitext(name)
        if os.path.isfile(path) and extension.lower() in IMAGE_EXTENSIONS:
            images.setdefault(stem, []).append(path)
    return images


def _discover_dataset_pairs(test_root):
    pairs = []
    for dataset_name in sorted(os.listdir(test_root)):
        dataset_dir = os.path.join(test_root, dataset_name)
        ir_dir = os.path.join(dataset_dir, 'ir')
        vi_dir = os.path.join(dataset_dir, 'vi')
        if not (os.path.isdir(ir_dir) and os.path.isdir(vi_dir)):
            continue

        visible_images = _index_images(vi_dir)
        for ir_name in sorted(os.listdir(ir_dir)):
            ir_path = os.path.join(ir_dir, ir_name)
            stem, extension = os.path.splitext(ir_name)
            if (
                not os.path.isfile(ir_path)
                or extension.lower() not in IMAGE_EXTENSIONS
            ):
                continue
            visible_candidates = visible_images.get(stem, [])
            if not visible_candidates:
                print(f"[WARN] Missing visible image for {dataset_name}/{ir_name}")
                continue
            pairs.append(
                (
                    dataset_name,
                    os.path.basename(visible_candidates[0]),
                    ir_path,
                    visible_candidates[0],
                )
            )
    return pairs


def val_fusion_detection(model, test_root, output_root):
    pairs = _discover_dataset_pairs(test_root)
    if not pairs:
        raise RuntimeError(f"No ir/vi image pairs found under: {test_root}")

    print(f"Total fusion pairs: {len(pairs)}")
    start = time.time()
    saved = 0

    for index, (dataset_name, output_name, ir_path, vi_path) in enumerate(
        pairs, start=1
    ):
        print(
            f"| Total test:{len(pairs)} | Now test:"
            f"{dataset_name}/{output_name} | Progress:{index}/{len(pairs)}"
        )

        vi_img = cv2.imread(vi_path)
        ir_img = cv2.imread(ir_path)
        if vi_img is None or ir_img is None:
            print(
                f"[WARN] Failed to read image pair: "
                f"{ir_path} | {vi_path}"
            )
            continue
        vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)

        vi_tensor = transforms.ToTensor()(vi_img).unsqueeze(0).cuda()
        ir_tensor = transforms.ToTensor()(ir_img).unsqueeze(0).cuda()

        with torch.no_grad():
            fused, _, _, _ = model.forward(
                ir_tensor,
                vi_tensor,
                None,
                None,
            )

        fusion_img = fused[0].detach().cpu().numpy()
        fusion_img = np.transpose(fusion_img, (1, 2, 0))
        fusion_img = np.clip(fusion_img * 255.0, 0, 255).astype(np.uint8)
        fusion_img = cv2.cvtColor(fusion_img, cv2.COLOR_RGB2BGR)

        dataset_output = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
        output_path = os.path.join(dataset_output, output_name)
        extension = os.path.splitext(output_name)[1]
        success, encoded = cv2.imencode(extension, fusion_img)
        if not success:
            raise RuntimeError(f"Failed to encode output image: {output_path}")
        encoded.tofile(output_path)
        saved += 1

    end = time.time()
    print(f"Saved fusion images: {saved}")
    print(f"Total test time: {end - start:.4f} s")
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument(
        '--test_root',
        type=str,
        required=True,
        help='root containing dataset folders with ir/vi subfolders',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='root for fused images',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoint_50epoch/model_galker_q_tr_invnorm.pth',
        help='joint fusion-detection checkpoint',
    )
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('| Testing fusion model on GPU #%d with pytorch' % (args.gpu))


    class Config():

        pretrained = False
        freeze_bn = True

        fpn_out_channels = 256
        use_p5 = True


        class_num = 6
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = True

        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        score_threshold = 0.3
        nms_iou_threshold = 0.4
        max_detection_boxes_num = 300


    model = Network(
        None,
        mode='inference',
        config=Config,
    ).cuda()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_state = checkpoint
    if isinstance(checkpoint, dict):
        checkpoint_state = checkpoint.get('state_dict', checkpoint)
    has_sfdfuse_weights = any(
        key.startswith('enhance_net.net.') or key.startswith('module.enhance_net.net.')
        for key in checkpoint_state.keys()
    )
    if not has_sfdfuse_weights:
        raise RuntimeError(
            "The checkpoint does not contain SFDFuse weights. "
            "Use a joint checkpoint trained with the SFDFuse fusion branch."
        )
    load_result = model.load_state_dict(checkpoint, strict=False)
    print(
        f"| Checkpoint loaded: {args.checkpoint} "
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    model.eval()


    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    print('| Model Load Done!')


    val_fusion_detection(model, args.test_root, args.output_dir)
