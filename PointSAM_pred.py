from point_sam.build_model import build_point_sam
import numpy as np
import torch.nn as nn
import torch
from torkit3d.ops.sample_farthest_points import sample_farthest_points
from torkit3d.nn.functional import batch_index_select
import math
import torch.nn.functional as F


def remove_zero_masks(masks, scores):
    # 通过检查每行是否全为零，返回一个布尔掩码
    non_zero_mask_flags = torch.any(masks != 0, dim=1)
    # 过滤掉全零的掩码
    filtered_masks = masks[non_zero_mask_flags]
    filterd_scores = scores[non_zero_mask_flags]
    return filtered_masks, filterd_scores

def compute_mask_iou(mask1, mask2):
    """Compute IoU between a single mask and multiple masks.

    Args:
        mask1: [1, N] - The single mask.
        masks: [M, N] - The masks to compare against.

    Returns:
        torch.Tensor: [M] - IoU values for each mask.
    """
    # 计算交集和并集
    intersection = torch.sum(mask1 & mask2, dim=1)
    union = torch.sum(mask1 | mask2, dim=1)

    # 避免除以零
    iou = intersection / union.clamp(min=1e-6)  # 使用 clamp 来防止除以零
    return iou

def is_foreground_in_mask(m1, m2):
    # 仅选择 m1 前景部分进行比较
    return torch.all(m2[m1 == 1] == 1)


def count_zero_masks(masks):
    # 判断每个掩码是否全为零
    zero_mask_flags = torch.all(masks == 0, dim=1)
    # 统计全零掩码的个数
    return torch.sum(zero_mask_flags).item()

    
class PointSAM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pointsam = build_point_sam(args.ckpt, 256, 32)
        self.pointsam.eval()
    
    def forward(self, feature_coords, features, pred = None, pred_gt = None, offest = None, query_pred_low= None):
       '''
       feature_coords: point cloud coords, shape N_pt x 3
       features: point cloud features, e.g. rgb, depth, N_pt x g
       pred: prediction logits, shape 1 x num_classes x N
       pred_gt: prediction ground truth, shape N
       '''

       #return self._forward_fps(feature_coords, features, pred_gt = pred_gt)
       #return self._forward_topk(feature_coords, features, pred_logits=pred, pred_gt = pred_gt)
       return self._forward_multi_class(feature_coords, features, pred_gt = pred_gt, offest=offest)

    
    def _forward_topk(self, feature_coords, features, pred_logits = None, pred_gt = None):
        feature_coords = feature_coords.unsqueeze(0) 
        features = features.unsqueeze(0) 
        norm_feature_coords = self.normalize_coords(feature_coords) # 1 x N_pt x 3

        self.pointsam.set_pointcloud(norm_feature_coords, features)

        pred_val,pred_imd = pred_logits.max(dim = 0) # N

        mask = pred_gt.bool()
        positive_index = None
        negetive_index = None
        count_ones = torch.sum(mask.int() == 1).item()
        count_zeros = torch.sum(mask.int() == 0).item()
        with torch.no_grad():
            if count_ones != 0:
                val, indices = torch.topk(pred_val * pred_gt.int(), min(50, count_ones))
                positive_index = norm_feature_coords.squeeze(0)[indices].unsqueeze(0)
                if count_ones-positive_index.shape[1] > 0:
                    positive_index2 = self.fps(norm_feature_coords.squeeze(0)[mask].unsqueeze(0), min(50, count_ones-positive_index.shape[1]))
                    positive_index = torch.cat([positive_index, positive_index2], dim = 1)
                positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
            if count_zeros != 0:
                val, indices = torch.topk(pred_val * (1 -pred_gt.int()), min(50, count_zeros))
                negetive_index = norm_feature_coords.squeeze(0)[indices].unsqueeze(0)
                if count_zeros - negetive_index.shape[1] > 0:
                    negetive_index2 = self.fps(norm_feature_coords.squeeze(0)[~mask].unsqueeze(0), min(50, count_zeros-negetive_index.shape[1]))
                    negetive_index = torch.cat([negetive_index, negetive_index2], dim = 1)
                negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
    
        if count_ones != 0 and count_zeros !=0:
                prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
                prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
        
        elif count_ones != 0:
            prompt_coords = positive_index
            prompt_labels = positive_label
        elif count_zeros !=0:
            prompt_coords = negetive_index
            prompt_labels = negetive_label
        else:
            return torch.zeros((2, features.shape[1])).cuda()
        mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt
        scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
        #    #return_mask = mask[0][torch.argmax(scores[0])]
        logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0) # 1, N
        return torch.cat([1-logits, logits])

    

    def _forward_fps_low(self, feature_coords, features, pred = None, pred_gt = None,query_pred_low= None):
       feature_coords = feature_coords.unsqueeze(0) # 1 x N_pt x 3
       features = features.unsqueeze(0) # 1 x N_pt x g 
       # the coords must be normalized to [-1, 1], 先试着归一化一下。
       norm_feature_coords = self.normalize_coords(feature_coords) # 1 x N_pt x 3

       self.pointsam.set_pointcloud(norm_feature_coords, features)
       mask = pred_gt.bool()
       positive_index = None
       negetive_index = None
       count_ones = torch.sum(mask.int() == 1).item()
       count_zeros = torch.sum(mask.int() == 0).item()

       with torch.no_grad():
         if count_ones != 0:
            positive_index = self.fps(norm_feature_coords.squeeze(0)[mask].unsqueeze(0), min(100, count_ones)) # 1 x 30 x 3
            center = self.find_center_of_point_cloud(norm_feature_coords.squeeze(0)[mask].unsqueeze(0)) # 1 x 1 x 3
            # positive_index = self.fps(fg_coords.unsqueeze(0), min(20, fg_count))
            # center = self.find_center_of_point_cloud(fg_coords.unsqueeze(0))
            positive_index = torch.cat([positive_index, center], dim = 1)
            # positive_index = fg_coords.unsqueeze(0)
            positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
         if count_zeros != 0:
            negetive_index = self.fps(norm_feature_coords.squeeze(0)[~mask].unsqueeze(0), min(count_zeros, 100)) # x 1 x 100 x 3
            # negetive_index = self.fps(bg_coords.unsqueeze(0), min(20, bg_count))
            # negetive_index = bg_coords.unsqueeze(0)
            negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
    
       if count_ones != 0 and count_zeros !=0:
            prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
            prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
        
       elif count_ones != 0:
           prompt_coords = positive_index
           prompt_labels = positive_label
       else:
           prompt_coords = negetive_index
           prompt_labels = negetive_label
       mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt
       scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
       #return_mask = mask[0][torch.argmax(scores[0])]
       logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0)
       return torch.cat([1-logits, logits], dim = 0)

       
    
    def panduan(self, qry_feat, sup_feat, qry_label, sup_label, qry_offset, sup_offset):
        sup_feat = sup_feat.cuda()
        qry_feat = qry_feat.cuda()
        qry_label = qry_label.cuda()
        sup_label = sup_label.cuda()
        
        qry_offset_cpu = qry_offset[:-1].long().cpu()
        qry_feat_split = torch.tensor_split(qry_feat, qry_offset_cpu)
        qry_label_split = torch.tensor_split(qry_label, qry_offset_cpu)

        sup_offset_cpu = sup_offset[:-1].long().cpu()
        sup_feat_split = torch.tensor_split(sup_feat, sup_offset_cpu)
        sup_label_split = torch.tensor_split(sup_label, sup_offset_cpu)

        sup_prototypes = []
        for i, (feat,label) in enumerate(zip(sup_feat_split, sup_label_split)):
            feat_feat = (feat[:, 3:6].contiguous() * label.unsqueeze(1).int()).unsqueeze(0)
            feat_coord = feat[:, :3].contiguous().unsqueeze(0)

            norm_feat_coord = self.normalize_coords(feat_coord)
            self.pointsam.set_pointcloud(norm_feat_coord, feat_feat)

            supp_embeddings = self.pointsam.pc_embeddings # [1, 512, 256]
            sup_prototypes.append(supp_embeddings.mean(dim = 1))

        proto_for_class1 = torch.mean(
            torch.cat(sup_prototypes[0:5], dim = 0),
            dim = 0,
            keepdim=True
        )

        proto_for_class2 = torch.mean(
            torch.cat(sup_prototypes[5:10], dim = 0),
            dim = 0,
            keepdim=True
        )

        all_proto = torch.cat([proto_for_class1, proto_for_class2], dim = 0)
        
        for i, (feat,label) in enumerate(zip(qry_feat_split, qry_label_split)):
            feat_feat = (feat[:, 3:6].contiguous()).unsqueeze(0)
            feat_coord = feat[:, :3].contiguous().unsqueeze(0)

            norm_feat_coord = self.normalize_coords(feat_coord)
            self.pointsam.set_pointcloud(norm_feat_coord, feat_feat)

            qry_embeddings = self.pointsam.pc_embeddings # [1, 512, 256]

            print(torch.unique(label))

            sim = F.cosine_similarity(qry_embeddings, all_proto.unsqueeze(1), dim = 2)

            threshold = 0.7

            high_sim_1 = sim[0,:] > threshold  # 类别1相似度高的点
            high_sim_2 = sim[1,:] > threshold  # 类别2相似度高的点

            # 分别统计点数
            count_class1 = high_sim_1.sum().item()
            count_class2 = high_sim_2.sum().item()

            print("count1", count_class1)
            print("count2", count_class2)







    def _forward_multi_class(self, feature_coords, features, pred = None, pred_gt = None, offest = None):
        cutoff = offest[0]
        feature_coords_split0, feature_coords_split1 = feature_coords[:cutoff], feature_coords[cutoff:]
        feature_split0, feature_split1 = features[:cutoff], features[cutoff:]
        pred_gt_split0, pred_gt_split1 = pred_gt[:cutoff], pred_gt[cutoff:]

        feature_list = [feature_split0, feature_split1]
        coords_list = [feature_coords_split0, feature_coords_split1]
        pred_list = [pred_gt_split0, pred_gt_split1]
        prob_maps = []
        for i, (coords, feature, pred_) in enumerate(zip(coords_list, feature_list, pred_list)):
             if coords.shape[0] < 512:
                 prob_maps.append(torch.zeros(size=(3, coords.shape[0]), device=coords.device))
                 continue
             coords = coords.unsqueeze(0) # 1 x N_pt x 3
             feature = feature.unsqueeze(0) # 1 x N_pt x g 
             norm_coords = self.normalize_coords(coords) # 1 x N_pt x 3
             self.pointsam.set_pointcloud(norm_coords, feature)
             indices_0 = (pred_ == 0).nonzero(as_tuple=True)[0]
             prob_map_t = []
             for j in range(2):
                indices_1 = (pred_ == (j+1)).nonzero(as_tuple=True)[0]
                prob_map = self._forward_with_label(norm_coords, indices_0, indices_1)
                prob_map_t.append(prob_map)
             # prob_map_t包含对背景、前景1、前景2的预测概率
             prob_map1, prob_map2 = prob_map_t # 2 X N
             #bg_prob = (prob_map1[[0]] * prob_map2[[0]])
             bg_prob = torch.min(prob_map1[[0]], prob_map2[[0]]) # 
             res = torch.cat([bg_prob, prob_map1[[1]], prob_map2[[1]]], dim = 0)
             #res = F.softmax(res, dim = 0)
             #res = res / res.sum(dim = 0)
             prob_maps.append(res)
        return torch.cat(prob_maps, dim = 1)
    

    def _forward_with_label(self, coords, indices0, indices1):
        positive_index = None
        negetive_index = None
        count_ones = indices1.shape[0]
        count_zeros = indices0.shape[0]
        device = coords.device
        if count_ones ==0 and count_zeros == 0:
            return torch.zeros(size=(2, coords.shape[1]), device=device)
        with torch.no_grad():
            if count_ones != 0:
                positive_index = self.fps(coords.squeeze(0)[indices1].unsqueeze(0), min(100, count_ones)) # 1 x 30 x 3
                center = self.find_center_of_point_cloud(coords.squeeze(0)[indices1].unsqueeze(0)) # 1 x 1 x 3
                # positive_index = self.fps(fg_coords.unsqueeze(0), min(20, fg_count))
                # center = self.find_center_of_point_cloud(fg_coords.unsqueeze(0))
                positive_index = torch.cat([positive_index, center], dim = 1)
                # positive_index = fg_coords.unsqueeze(0)
                positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
            if count_zeros != 0:
                negetive_index = self.fps(coords.squeeze(0)[indices0].unsqueeze(0), min(count_zeros, 100)) # x 1 x 100 x 3
                # negetive_index = self.fps(bg_coords.unsqueeze(0), min(20, bg_count))
                # negetive_index = bg_coords.unsqueeze(0)
                negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
        
        if count_ones != 0 and count_zeros !=0:
                prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
                prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
            
        elif count_ones != 0:
            prompt_coords = positive_index
            prompt_labels = positive_label
        else:
            prompt_coords = negetive_index
            prompt_labels = negetive_label
        mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt # B x 1 x N_pt
        mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True, prompt_masks=logits[:,1,:])
        scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
        #return_mask = mask[0][torch.argmax(scores[0])]
        logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0)
        return torch.cat([1-logits, logits], dim = 0)

    def _forward_dpc(self, feature_coords, features, pred = None, pred_gt = None):
        feature_coords = feature_coords.unsqueeze(0) # 1 x N_pt x 3
        features = features.unsqueeze(0) # 1 x N_pt x g 
        # the coords must be normalized to [-1, 1], 先试着归一化一下。
        norm_feature_coords = self.normalize_coords(feature_coords) # 1 x N_pt x 3

        self.pointsam.set_pointcloud(norm_feature_coords, features)
        mask = pred_gt.bool()
        positive_index = None
        negetive_index = None
        count_ones = torch.sum(mask.int() == 1).item()
        count_zeros = torch.sum(mask.int() == 0).item()
        with torch.no_grad():
            if count_ones !=0:
                positive_index = self.dpc(norm_feature_coords.squeeze(0)[mask].unsqueeze(0), min(200, count_ones))
                center = self.find_center_of_point_cloud(norm_feature_coords.squeeze(0)[mask].unsqueeze(0))
                positive_index = torch.cat([positive_index, center], dim = 1)
                # positive_index = fg_coords.unsqueeze(0)
                positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
            if count_zeros != 0:
                negetive_index = self.dpc(norm_feature_coords.squeeze(0)[~mask].unsqueeze(0), min(count_zeros, 200)) # x 1 x 100 x 3
                #negetive_index = norm_feature_coords.squeeze(0)[negetive_index].unsqueeze(0)
                # negetive_index = self.fps(bg_coords.unsqueeze(0), min(20, bg_count))
                # negetive_index = bg_coords.unsqueeze(0)
                negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
                
        if count_ones != 0 and count_zeros !=0:
                prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
                prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
            
        elif count_ones != 0:
            prompt_coords = positive_index
            prompt_labels = positive_label
        else:
            prompt_coords = negetive_index
            prompt_labels = negetive_label
        mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt
        scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
        #return_mask = mask[0][torch.argmax(scores[0])]
        logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0)
        return torch.cat([1-logits, logits], dim = 0)
        #return torch.cat([1-torch.sigmoid(logits)])
               

    def _forward_fps(self, feature_coords, features, pred = None, pred_gt = None):
         # reshape to fit the requirements of pointsam. i.e.,  tensors must have batch_size dimension
       feature_coords = feature_coords.unsqueeze(0) # 1 x N_pt x 3
       features = features.unsqueeze(0) # 1 x N_pt x g 
       # the coords must be normalized to [-1, 1], 先试着归一化一下。
       norm_feature_coords = self.normalize_coords(feature_coords) # 1 x N_pt x 3

       self.pointsam.set_pointcloud(norm_feature_coords, features)

    #    fore_indices, bg_indices, fg_count, bg_count = self.topk_foreground_background(pred_gt, 5, 1)

    #    fg_coords = norm_feature_coords.squeeze()[fore_indices]
    #    bg_coords = norm_feature_coords.squeeze()[bg_indices]
       # 前景选择postive prompt, 背景选择negative prompt
       mask = pred_gt.bool()
       positive_index = None
       negetive_index = None
       count_ones = torch.sum(mask.int() == 1).item()
       count_zeros = torch.sum(mask.int() == 0).item()
       with torch.no_grad():
         if count_ones != 0:
            positive_index = self.fps(norm_feature_coords.squeeze(0)[mask].unsqueeze(0), min(100, count_ones)) # 1 x 30 x 3
            center = self.find_center_of_point_cloud(norm_feature_coords.squeeze(0)[mask].unsqueeze(0)) # 1 x 1 x 3
            # positive_index = self.fps(fg_coords.unsqueeze(0), min(20, fg_count))
            # center = self.find_center_of_point_cloud(fg_coords.unsqueeze(0))
            positive_index = torch.cat([positive_index, center], dim = 1)
            # positive_index = fg_coords.unsqueeze(0)
            positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
         if count_zeros != 0:
            negetive_index = self.fps(norm_feature_coords.squeeze(0)[~mask].unsqueeze(0), min(count_zeros, 100)) # x 1 x 100 x 3
            # negetive_index = self.fps(bg_coords.unsqueeze(0), min(20, bg_count))
            # negetive_index = bg_coords.unsqueeze(0)
            negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
    
       if count_ones != 0 and count_zeros !=0:
            prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
            prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
        
       elif count_ones != 0:
           prompt_coords = positive_index
           prompt_labels = positive_label
       elif count_zeros !=0:
           prompt_coords = negetive_index
           prompt_labels = negetive_label
       else:
           return torch.zeros((2, features.shape[1])).cuda()
       mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt
       scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
    #    #return_mask = mask[0][torch.argmax(scores[0])]
       logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0) # 1, N
       return torch.cat([1-logits, logits])
       
    

    def _forward_fps_topk(self, feature_coords, features, pred = None, pred_gt = None):
       feature_coords = feature_coords.unsqueeze(0) # 1 x N_pt x 3
       features = features.unsqueeze(0) # 1 x N_pt x g 
       # the coords must be normalized to [-1, 1], 先试着归一化一下。
       norm_feature_coords = self.normalize_coords(feature_coords) # 1 x N_pt x 3

       self.pointsam.set_pointcloud(norm_feature_coords, features)
       # 前景选择postive prompt, 背景选择negative prompt
       mask = pred_gt.max(0)[1].bool()
       positive_index = None
       negetive_index = None
       count_ones = torch.sum(mask.int() == 1).item()
       count_zeros = torch.sum(mask.int() == 0).item()
       with torch.no_grad():
         if count_ones != 0:
            positive_index = sample_farthest_points(norm_feature_coords.squeeze(0)[mask].unsqueeze(0), min(150, count_ones)) # 1 x 100
            center = self.find_center_of_point_cloud(norm_feature_coords.squeeze(0)[mask].unsqueeze(0)) # 1 x 1 x 3
            positive_val = self.find_topkcoords_from_prediction(positive_index.squeeze(0), pred_gt[1], norm_feature_coords.squeeze(0), 
                                                                min(positive_index.shape[1], 100)) # 1 x 50
            # positive_index = self.fps(fg_coords.unsqueeze(0), min(20, fg_count))
            # center = self.find_center_of_point_cloud(fg_coords.unsqueeze(0))
            positive_index = torch.cat([positive_val, center], dim = 1)
            # positive_index = fg_coords.unsqueeze(0)
            positive_label = torch.ones(size=positive_index.shape[:2], device = positive_index.device)
         if count_zeros != 0:
            negetive_index = sample_farthest_points(norm_feature_coords.squeeze(0)[~mask].unsqueeze(0), min(count_zeros, 150)) # x 1 x 100 x 3
            # negetive_index = self.fps(bg_coords.unsqueeze(0), min(20, bg_count))
            # negetive_index = bg_coords.unsqueeze(0)
            negetive_index = self.find_topkcoords_from_prediction(negetive_index.squeeze(0), pred_gt[0], norm_feature_coords.squeeze(0), 
                                                                min(negetive_index.shape[1], 100)) # 1 x 50
            negetive_label = torch.zeros(size=negetive_index.shape[:2], device = negetive_index.device)
    
       if count_ones != 0 and count_zeros !=0:
            prompt_coords = torch.cat([positive_index, negetive_index], dim = 1) # 1 x (30 + 100) x 3
            prompt_labels = torch.cat([positive_label, negetive_label], dim = 1) # 1 x (30 + 100)
        
       elif count_ones != 0:
           prompt_coords = positive_index
           prompt_labels = positive_label
       else:
           prompt_coords = negetive_index
           prompt_labels = negetive_label
       mask, scores, logits = self.pointsam.predict_masks(prompt_coords, prompt_labels, multimask_output=True) # B x 1 x N_pt
       scores = scores.squeeze(0) / scores.squeeze(0).sum(dim=0)
       #return_mask = mask[0][torch.argmax(scores[0])]
       logits = torch.sigmoid(torch.sum(logits[0] * scores.unsqueeze(1), dim = 0)).unsqueeze(0)
       return torch.cat([1-logits, logits], dim = 0)

       

    def normalize_coords(self, coords):
        # corrds shape 1 x N x 3, normalize coords to [-1,1]
        xyz = coords.squeeze(0)
        xyz = xyz.cpu().numpy()
        shift = xyz.mean(0)
        scale = np.linalg.norm(xyz - shift, axis=-1).max()
        xyz = (xyz - shift) / scale
        return torch.from_numpy(xyz).to(coords.device).unsqueeze(0)
    

    def fps(self, points: torch.Tensor, num_samples: int):
        """A wrapper of farthest point sampling (FPS).

        Args:
            points: [B, N, 3]. Input point clouds.
            num_samples: int. The number of points to sample.

        Returns:
            torch.Tensor: [B, num_samples, 3]. Sampled points.
        """
        idx = sample_farthest_points(points, num_samples)
        sampled_points = batch_index_select(points, idx, dim=1)
        return sampled_points
    

    def find_center_of_point_cloud(self, point_cloud):
        """
        找到形状为 1 x N x 3 的点云的中心点（质心）
        :param point_cloud: Tensor, 大小为 (1, N, 3)，表示 1 个点云，包含 N 个点，每个点有 (x, y, z) 三个坐标
        :return: Tensor，表示点云的中心点，大小为 (1, 3)
        """
        # 去掉第一维，变为 (N, 3)
        point_cloud = point_cloud.squeeze(0)

        # 对所有点的坐标分别取平均值，得到质心坐标
        center = torch.mean(point_cloud, dim=0, keepdim=True)

        # 恢复到 (1, 3) 形状
        center = center.unsqueeze(0)

        return center
    

    def find_topkcoords_from_prediction(self, coord_indices, pred, coords, k):
        # predictions是经过softmax归一化后的。单通道

        val = pred[coord_indices] # 取出
        topk_indices = torch.topk(val, k).indices # 
        topk_val = coord_indices[topk_indices]
        return coords[topk_val].unsqueeze(0)
    

    def dpc(self, coords, k):
        # coords has shape 1 x N X 3
        device = coords.device
        N = coords.shape[1]
        dist = torch.cdist(coords, coords)
        # ner = math.ceil(N * 0.1 /100)
        ner = 3
        # dist_nearest, index_nearest = torch.topk(dist, k=min(ner, N), dim=-1, largest=False)
        density = ((-((dist/ner) ** 2)).exp()).mean(dim = -1)
        # density = density + torch.rand(
        #     density.shape, device=density.device, dtype=density.dtype) * 1e-6
        
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(coords.dtype)
        dist_max = dist.flatten(1).max(dim=-1)[0][:, None, None]
        distx, index_parent = (dist * mask + dist_max * (1 - mask)).min(dim=-1)
        # select clustering center according to score
        score = distx * density # 1 x N
        _, index_down = torch.topk(score, k=k, dim=-1) # 1 X K下标
        #return index_down.squeeze(0)

        return coords.squeeze(0)[index_down.squeeze(0)].unsqueeze(0)
    


    def forward_mask_proposals(self, feature, coord, num_seeds = 100, voxel_index = None, crop_index = None, within_mask = None, flag = False):
        """
        Generating mask proposals, follow AAAI
        

        feature: Support/Query feature. Shape N_pt x 3
        coord: Support/Query coord. Shape N_pt x 3
        num_seeds: initial 
        """

        if feature.shape[0] < 512:
            return None, None
        
        feature = feature.unsqueeze(0)
        coord = coord.unsqueeze(0)
        norm_coord = self.normalize_coords(coord)
        with torch.no_grad():
            if flag:
                self.pointsam.set_pointcloud(norm_coord, feature)
        
        if voxel_index is None:
            fps_seeds = self.fps(
                norm_coord,
                num_samples=min(num_seeds, feature.shape[1])
            ).squeeze(0).float() # num_seeds, 3
        else:
            fps_seeds = norm_coord[0,voxel_index, :]
            fps_seeds = fps_seeds[crop_index, :]
        # for each fps_seeds, produce three masks and applying nms to filter them
        masks_list = []
        scores_list = []
        for seed in fps_seeds:
            device = seed.device
            label = torch.ones(
                1,
                dtype=torch.bool,
                device = device).unsqueeze(0) # 1 x 1
            seed_ = seed.unsqueeze(0).unsqueeze(0) # 1 x 1 x 3

            mask, scores, logits = self.pointsam.predict_masks(
                seed_, label, multimask_output=True) # mask has shape 1 x 3 x N, scores has shape 1 x 3
            
            masks_list.append(
                mask.squeeze(0)
            )

            scores_list.append(
                scores.squeeze(0)
            )
        
        mask_proposal = torch.cat(
            masks_list, dim = 0
        ) # Shape. (num_seeds x 3) x N


        confidence = torch.cat(
            scores_list, dim = 0
        ) # Shape. (num_seeds x 3)

        # device = fps_seeds.device
        # label = torch.ones(
        #     fps_seeds.shape[0],
        #     dtype=torch.bool,
        #     device = device).unsqueeze(0)
        
        # mask_proposal, confidence, logits = self.pointsam.predict_masks(
        #         fps_seeds.unsqueeze(0), label, multimask_output=False) # mask has shape 1 x 3 x N, scores has shape 1 x 3

        mask_proposal, confidence = remove_zero_masks(mask_proposal.squeeze(0), confidence.squeeze(0))

        indices = (confidence >= 0.8).nonzero(as_tuple=True)[0]

        mask_proposal = mask_proposal[indices, :]

        confidence = confidence[indices]

        sorted_scores, sorted_indices = torch.sort(
            confidence, descending=True
        )
        
        indices_to_keep = []
        while sorted_indices.shape[0] > 0:
            idx = sorted_indices[0] # 当前分数最高下标
            indices_to_keep.append(idx)

            # 计算idx与其它所有剩余masks的iou
            mask_idx = mask_proposal[[idx]] # 1 x N
            pair_wise_IoUs = compute_mask_iou(
                mask_idx.unsqueeze(0), mask_proposal[sorted_indices[1:]]
            ).squeeze(0) # T

            idxs_keep = torch.where(pair_wise_IoUs <= 0.3)[0] + 1
            sorted_indices = sorted_indices[idxs_keep]
        
        indices_to_keep_tensor = torch.tensor(indices_to_keep, dtype=torch.long)

        final_mask_proposals = mask_proposal[indices_to_keep_tensor, :].int() # 保存下来的

        confidence = confidence[indices_to_keep_tensor]
        # #final_mask_proposals = mask_proposal

        # # 消除重叠的混合区域
        # areas = final_mask_proposals.sum(dim=1)

        # mask_indices = torch.argsort(areas, descending=True)
        # sorted_mask_collection = final_mask_proposals[mask_indices]

        # confidence = confidence[mask_indices]

        # M, N = sorted_mask_collection.shape
        # index_mask = torch.zeros(N, dtype=torch.long, device='cuda') + M  # 初始化为 N

        # # 更新索引掩膜
        # for i in range(M):
        #     index_mask[sorted_mask_collection[i] == 1] = i

        # # 创建一热编码掩膜
        # one_hot_masks = torch.nn.functional.one_hot(index_mask)[:, :M]

        # #调整维度顺序
        # final_mask_proposals = one_hot_masks.permute(1, 0)  # 变为 (M, N)
        return final_mask_proposals, confidence







            








