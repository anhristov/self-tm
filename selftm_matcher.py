import numpy as np
import torch
from torch import nn
import selftm_model as selftm


class SelfTMMatcher(nn.Module):
    def __init__(self, _arch, _downscales, _drop_path_rate, _layer_scale_init_value):
        super().__init__()
        self.downscales = _downscales
        self.arch = _arch
        self.drop_path_rate = _drop_path_rate
        self.layer_scale_init_value = _layer_scale_init_value

        self.backbone, self.representations_dim = selftm.__dict__[self.arch](drop_path_rate=self.drop_path_rate, layer_scale_init_value=self.layer_scale_init_value, )

    @staticmethod
    def soft_argmax_2d(_input, _beta=1000):
        _c, _h, _w = _input.shape

        _input = _input.reshape(_c, _h * _w)
        _input = nn.functional.softmax(_beta * _input, dim=-1)

        _indices_c, _indices_r = np.meshgrid(np.linspace(0, 1, _w), np.linspace(0, 1, _h), indexing='xy')

        _indices_r = torch.tensor(np.reshape(_indices_r, (-1, _h * _w)), dtype=torch.float32).cuda()
        _indices_c = torch.tensor(np.reshape(_indices_c, (-1, _h * _w)), dtype=torch.float32).cuda()

        _result_r = torch.sum((_h - 1) * _input * _indices_r, dim=-1)
        _result_c = torch.sum((_w - 1) * _input * _indices_c, dim=-1)

        _result = torch.stack([_result_r, _result_c], dim=-1)  # [y, x]

        return _result

    def match_template(self, _query, _template):

        def compute_search_region_coords(_current_templ_abs_coords, _current_padding, _h, _w):
            _current_templ_y = round(_current_templ_abs_coords[0] * _h)
            _current_templ_x = round(_current_templ_abs_coords[1] * _w)
            _current_templ_y1 = max(0, round(_current_templ_y - _current_padding))
            _current_templ_x1 = max(0, round(_current_templ_x - _current_padding))
            _current_templ_y2 = min(round(_current_templ_y + _current_padding * 2), _h)
            _current_templ_x2 = min(round(_current_templ_x + _current_padding * 2), _w)

            return _current_templ_y1, _current_templ_x1, _current_templ_y2, _current_templ_x2

        # Compute features of whole batch of images
        _query_features_hierarchy = self.backbone(_query)
        _template_features_hierarchy = self.backbone(_template)

        # Compute multipliers
        _multipliers = []
        for _i in range(len(self.downscales[::-1])):
            if _i == 0:
                _multipliers.append(self.downscales[::-1][_i])
            else:
                _multipliers.append(_multipliers[_i - 1] * self.downscales[::-1][_i])

        _pred_abs_coords = []
        _pred_confs = []
        for _fmap_id in range(len(_query_features_hierarchy)):
            # Features
            _query_features = _query_features_hierarchy[_fmap_id]
            _templates_features = _template_features_hierarchy[_fmap_id]

            # Compute template feature coordinates
            _, _, _img_feat_h, _img_feat_w = _query_features.shape

            #########################################################################
            # Computing template matching loss
            #########################################################################
            # Loop over all templates
            _template_matching_loss = 0
            _pred_templ_abs_coords = []
            _pred_templ_coords = []
            _featuremaps_visualization = []

            _result1 = torch.nn.functional.conv2d(_query_features, _templates_features, bias=None, stride=1, padding='same')
            _result2 = torch.sqrt(torch.sum(_templates_features ** 2) * torch.nn.functional.conv2d(_query_features ** 2, torch.ones_like(_templates_features), bias=None, stride=1, padding='same'))
            _TM_CCORR_NORMED = (_result1 / _result2).squeeze(0).squeeze(0)  # Working

            # Compute template's predicted location inside the distance map
            if _fmap_id > 0:
                # Compute search region coords based on first location (deepest layer featuremap)
                _multiplier_padding = 1 * _multipliers[_fmap_id - 1]
                _first_pred_templ_abs_coords = _pred_abs_coords[0]
                _first_pred_templ_y1, _first_pred_templ_x1, _first_pred_templ_y2, _first_pred_templ_x2 = compute_search_region_coords(_first_pred_templ_abs_coords, _multiplier_padding, _img_feat_h, _img_feat_w)

                # Compute search region coords based on previous location
                _padding = 1 * self.downscales[::-1][_fmap_id - 1]  # Pad searched region with 1 cell (pixel) from previous feature map (which is equal to the downscales factor)
                _prev_pred_templ_abs_coords = _pred_abs_coords[_fmap_id - 1]
                _prev_pred_templ_y1, _prev_pred_templ_x1, _prev_pred_templ_y2, _prev_pred_templ_x2 = compute_search_region_coords(_prev_pred_templ_abs_coords, _padding, _img_feat_h, _img_feat_w)
                _search_templ_y1 = max(_first_pred_templ_y1, _prev_pred_templ_y1)
                _search_templ_x1 = max(_first_pred_templ_x1, _prev_pred_templ_x1)
                _search_templ_y2 = min(_first_pred_templ_y2, _prev_pred_templ_y2)
                _search_templ_x2 = min(_first_pred_templ_x2, _prev_pred_templ_x2)

                # Check
                assert _search_templ_y2 - _search_templ_y1 > 0
                assert _search_templ_x2 - _search_templ_x1 > 0

                _pred_yx = self.soft_argmax_2d(_TM_CCORR_NORMED[_search_templ_y1:_search_templ_y2, _search_templ_x1:_search_templ_x2].unsqueeze(dim=0))  # [y, x]
                _pred_yx[0][0] += _search_templ_y1
                _pred_yx[0][1] += _search_templ_x1

                assert _first_pred_templ_y1 <= _pred_yx[0][0] < _first_pred_templ_y2
                assert _first_pred_templ_x1 <= _pred_yx[0][1] < _first_pred_templ_x2
            else:
                _pred_yx = self.soft_argmax_2d(_TM_CCORR_NORMED.unsqueeze(dim=0))  # [y1, x1]

            # Compute absolute predicted template coordinates
            _prex_y_abs = _pred_yx[0][0].item() / _img_feat_h
            _prex_x_abs = _pred_yx[0][1].item() / _img_feat_w

            _pred_abs_coords.append([_prex_y_abs, _prex_x_abs])
            _pred_confs.append(_TM_CCORR_NORMED[round(_pred_yx[0][0].item()), round(_pred_yx[0][1].item())].detach().cpu().item())
            #########################################################################

        return _pred_abs_coords, _pred_confs