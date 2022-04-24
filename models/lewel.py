# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from math import cos, pi
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss
from utils import init


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='normal'):
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ObjectNeck(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 num_layers=1,
                 scale=1.,
                 l2_norm=True,
                 num_heads=8, 
                 norm_layer=None,
                 **kwargs):
        super(ObjectNeck, self).__init__()

        self.scale = scale
        self.l2_norm = l2_norm
        assert l2_norm
        self.num_heads = num_heads

        hid_channels = hid_channels or in_channels
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
    
    def init_weights(self, init_linear='kaiming'):
        self.proj.init_weights(init_linear)
        self.proj_obj.init_weights(init_linear)
    
    def forward(self, x):
        b, c, h, w = x.shape

        # flatten and projection
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(2)
        x = x.flatten(2)    # (bs, c, h*w)
        z = self.proj(torch.cat([x_pool, x], dim=2))    # (bs, k, 1+h*w)
        z_g, obj_attn = torch.split(z, [1, x.shape[2]], dim=2)  # (bs, nH*k, 1), (bs, nH*k, h*w)

        # do attention according to obj attention map
        obj_attn = F.normalize(obj_attn, dim=1) if self.l2_norm else obj_attn
        obj_attn /= self.scale
        obj_attn = F.softmax(obj_attn, dim=2)
        obj_attn = obj_attn.view(b, self.num_heads, -1, h*w)
        x = x.view(b, self.num_heads, -1, h*w)
        obj_val = torch.matmul(x, obj_attn.transpose(3, 2))    # (bs, nH, c//Nh, k)
        obj_val = obj_val.view(b, c, obj_attn.shape[-2])    # (bs, c, k)

        # projection
        obj_val = self.proj_obj(obj_val)    # (bs, k, k)

        return z_g, obj_val # (bs, k, 1), (bs, k, k), where the second dim is channel
    
    def extra_repr(self) -> str:
        parts = []
        for name in ["scale", "l2_norm", "num_heads"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)


class EncoderObj(nn.Module):
    def __init__(self, base_encoder, hid_dim, out_dim, norm_layer=None, num_mlp=2,
                 scale=1., l2_norm=True, num_heads=8):
        super(EncoderObj, self).__init__()
        self.backbone = base_encoder(norm_layer=norm_layer, with_avgpool=False)
        in_dim = self.backbone.out_channels
        self.neck = ObjectNeck(in_channels=in_dim, hid_channels=hid_dim, out_channels=out_dim,
                               norm_layer=norm_layer, num_layers=num_mlp,
                               scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.neck.init_weights(init_linear='kaiming')

    def forward(self, im):
        out = self.backbone(im)
        out = self.neck(out)
        return out


class LEWELB_EMAN(nn.Module):
    def __init__(self, base_encoder, dim=256, m=0.996, hid_dim=4096, norm_layer=None, num_neck_mlp=2,
                 scale=1., l2_norm=True, num_heads=8, loss_weight=0.5, **kwargs):
        super().__init__()

        self.base_m = m
        self.curr_m = m
        self.loss_weight = loss_weight

        # create the encoders
        # num_classes is the output fc dimension
        self.online_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.target_net = EncoderObj(base_encoder, hid_dim, dim, norm_layer, num_neck_mlp,
                                     scale=scale, l2_norm=l2_norm, num_heads=num_heads)
        self.predictor = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor.init_weights()
        self.predictor_obj = MLP1D(dim, hid_dim, dim, norm_layer=norm_layer)
        self.predictor_obj.init_weights()

        # copy params from online model to target model
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)  # initialize
            param_tgt.requires_grad = False  # not update by gradient
    
    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return loss
        
    def loss_func(self, online, target):
        z_o, obj_o = online
        z_t, obj_t = target
        # instance-level loss
        loss_inst = self.mse_loss(self.predictor(z_o).squeeze(-1), z_t.squeeze(-1))
        # object-level loss
        b, c, n = obj_o.shape
        obj_o_pred = self.predictor_obj(obj_o).transpose(2, 1).reshape(b*n, c)
        obj_t = obj_t.transpose(2, 1).reshape(b*n, c)
        loss_obj = self.mse_loss(obj_o_pred, obj_t)
        # sum
        return loss_inst * self.loss_weight + loss_obj * (1 - self.loss_weight)
    
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        state_dict_ol = self.online_net.state_dict()
        state_dict_tgt = self.target_net.state_dict()
        for (k_ol, v_ol), (k_tgt, v_tgt) in zip(state_dict_ol.items(), state_dict_tgt.items()):
            assert k_tgt == k_ol, "state_dict names are different!"
            assert v_ol.shape == v_tgt.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_tgt:
                v_tgt.copy_(v_ol)
            else:
                v_tgt.copy_(v_tgt * momentum + (1. - momentum) * v_ol)

    def forward(self, im_v1, im_v2=None, **kwargs):
        """
        Input:
            im_v1: a batch of view1 images
            im_v2: a batch of view2 images
        Output:
            loss
        """
        # for inference, online_net.backbone model only
        if im_v2 is None:
            feats = self.online_net.backbone(im_v1)
            return F.adaptive_avg_pool2d(feats, 1).flatten(1)

        # compute online_net features
        proj_online_v1 = self.online_net(im_v1)
        proj_online_v2 = self.online_net(im_v2)

        # compute target_net features
        with torch.no_grad():  # no gradient to keys
            proj_target_v1 = [x.clone().detach() for x in self.target_net(im_v1)]
            proj_target_v2 = [x.clone().detach() for x in self.target_net(im_v2)]

        # loss. NOTE: the predction is moved to loss_func
        loss = self.loss_func(proj_online_v1, proj_target_v2) + \
            self.loss_func(proj_online_v2, proj_target_v1)

        return loss

    def extra_repr(self) -> str:
        parts = []
        for name in ["loss_weight"]:
            parts.append(f"{name}={getattr(self, name)}")
        return ", ".join(parts)


class LEWELB(LEWELB_EMAN):
    @torch.no_grad()
    def momentum_update(self, cur_iter, max_iter):
        """
        Momentum update of the target network.
        """
        # momentum anneling
        momentum = 1. - (1. - self.base_m) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2.0
        self.curr_m = momentum
        # parameter update for target network
        for param_ol, param_tgt in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_tgt.data = param_tgt.data * momentum + param_ol.data * (1. - momentum)


def get_lewel_model(model):
    """
    Args:
        model (str or callable):

    Returns:
        Model
    """
    if isinstance(model, str):
        model = {
            "LEWELB": LEWELB,
            "LEWELB_EMAN": LEWELB_EMAN,
        }[model]
    return model
