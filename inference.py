import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from selftm_matcher import SelfTMMatcher

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tensor_to_numpy_image(_tensor):
    return np.array(transforms.ToPILImage()(_tensor))[:, :, ::-1].copy()

def main():
    # Config
    arch = "selftm_base"    # "selftm_small", "selftm_large"
    last_checkpoint = './weights/selftm_base_imagenet_affine_hpatches_affine.pth'
    downscales = [3, 3, 3]
    drop_path_rate = 0.1
    layer_scale_init_value = 0.0

    # Input
    query_image_filename = './data/input_image.png'
    template_image_filename = './data/template.png'

    # Preprocessing
    preprocessing = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])])
    preprocessing_viz = transforms.Compose([transforms.ToTensor()])

    # Load the model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        gpu = torch.device('cuda')
        selftm_matcher = SelfTMMatcher(arch, downscales, drop_path_rate, layer_scale_init_value).cuda(gpu)
    else:
        selftm_matcher = SelfTMMatcher(arch, downscales, drop_path_rate, layer_scale_init_value)

    # Load the checkpoint
    if last_checkpoint is not None:
        ckpt = torch.load(last_checkpoint, map_location="cpu")
        state_dict = ckpt["model"]
        state_dict = {key.replace("module.", ""): value for (key, value) in state_dict.items()}
        selftm_matcher.load_state_dict(state_dict)

    # Put in evaluation mode
    selftm_matcher.eval()
    selftm_matcher.backbone.eval()

    query_image_np = cv2.imread(query_image_filename)
    template_image_np = cv2.imread(template_image_filename)

    query_image = Image.fromarray(cv2.cvtColor(query_image_np, cv2.COLOR_BGR2RGB))
    template_image = Image.fromarray(cv2.cvtColor(template_image_np, cv2.COLOR_BGR2RGB))

    if torch.cuda.is_available():
        query_image_preprocessed = preprocessing(query_image).unsqueeze(dim=0).cuda(gpu)
        template_image_preprocessed = preprocessing(template_image).unsqueeze(dim=0).cuda(gpu)
        template_image_preprocessed_viz = preprocessing_viz(template_image).unsqueeze(dim=0).cuda(gpu)
    else:
        query_image_preprocessed = preprocessing(query_image).unsqueeze(dim=0)
        template_image_preprocessed = preprocessing(template_image).unsqueeze(dim=0)
        template_image_preprocessed_viz = preprocessing_viz(template_image).unsqueeze(dim=0).cuda(gpu)

    # Match template
    _coords_abs, _confs = selftm_matcher.match_template(query_image_preprocessed, template_image_preprocessed)

    # Visualization
    _colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]   # [last: red, mid: green, first: blue]
    _img_h, _img_w, _ = query_image_np.shape
    _tmpl_h, _tmpl_w, _ = template_image_np.shape
    for _idx, _coord_abs in enumerate(_coords_abs):
        _x_abs = _coord_abs[1]
        _y_abs = _coord_abs[0]
        _center_x = _x_abs * _img_w
        _center_y = _y_abs * _img_h

        _x1 = round(_center_x - _tmpl_w / 2)
        _y1 = round(_center_y - _tmpl_h / 2)
        _x2 = round(_center_x + _tmpl_w / 2)
        _y2 = round(_center_y + _tmpl_h / 2)

        cv2.rectangle(query_image_np, (_x1, _y1), (_x2, _y2), _colors[_idx], 1)

    template_viz = tensor_to_numpy_image(template_image_preprocessed_viz[0])

    cv2.imshow('Output', query_image_np)
    cv2.imshow('Template', template_viz)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

