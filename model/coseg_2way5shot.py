from typing import Optional, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import wandb
from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer, CrossAttention
from PointSAM_pred import PointSAM
import torch_points_kernels as tp
import os
from util.logger import get_logger
from lib.pointops2.functions import pointops
from PointSAM_pred import compute_mask_iou
from scipy.optimize import linear_sum_assignment
from model.common import SlotAttention
from flash_attn import flash_attn_func
class COSeg(nn.Module):
    def __init__(self, args):
        super(COSeg, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        self.num_clustering = args.num_clustering
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(
            ignore_index=args.ignore_label
        )

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2**i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2**i) for i in range(args.num_layers)
        ]

        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1,
                    3: 2,
                    4: 3,
                    8: 4,
                    10: 5,
                    11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    2: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1,
                    3: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    10: 6,
                    12: 7,
                    13: 8,
                    14: 9,
                    19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    4: 2,
                    8: 3,
                    9: 4,
                    11: 5,
                    15: 6,
                    16: 7,
                    17: 8,
                    18: 9,
                    20: 10,
                }

        if self.main_process():
            self.logger = get_logger(args.save_path)

        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.args.classes // 2 + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path),
        )

        self.feat_dim = args.channels[2]

        self.visualization = args.vis

        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.kpconv = KPConvResBlock(
            self.feat_dim, self.feat_dim, 0.04, sigma=2
        )

        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )

        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim // 2, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        if self.args.data_name == "s3dis":
            agglayers = 2
        else:
            agglayers = 4

        print(f"use agglayers {agglayers}")
        self.agglayers = nn.ModuleList(
            [
                AggregatorLayer(
                    hidden_dim=self.feat_dim,
                    guidance_dim=0,
                    nheads=4,
                    attention_type="linear",
                )
                for _ in range(agglayers)
            ]
        )

        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * (self.n_way + 1), self.feat_dim
            )

        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )
        self.afha1 = 1.0 # 初始化

        self.mlp_for_class1 = MLPWithoutResidual(
            101, 100
        )
        self.mlp_for_class2 = MLPWithoutResidual(
            101,100
        )

        self.init_weights()

        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )


    def init_weights(self):
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
        support_proposals: Optional[torch.Tensor] = None,
        query_proposals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the COSEG model.

        Args:
            support_offset: Offset of each scene in support inputs. Shape (N_way*K_shot).
            support_x: Support point cloud input with shape (N_support, in_channels).
            support_y: Support masks with shape (N_support).
            query_offset: Offset of each scene in support inputs. Shape (N_way).
            query_x: Query point cloud input with shape (N_query, in_channels).
            query_y: Query labels with shape (N_query).
            epoch: Current epoch.
            support_base_y: Support base class labels with shape (N_support).
            query_base_y: Query base class labels with shape (N_query).
            sampled_classes: Sampled classes in current episode. Shape (N_way).

        Returns:
            query_pred: Predicted class logits for query point clouds. Shape: (1, n_way+1, N_query).
            loss: Forward loss value.
        """

        # get downsampled support features
        (
            support_feat,  # N_s, C
            support_x_low,  # N_s, 3
            support_offset_low,
            support_y_low,  # N_s
            _,
            support_base_y,  # N_s
            support_feat_wobkffn,
        ) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y
        )



        assert support_y_low.shape[0] == support_x_low.shape[0]
        # split support features and coords into list according to offset
        support_offset_low = support_offset_low[:-1].long().cpu()
        support_feat = torch.tensor_split(support_feat, support_offset_low) # Nway x kshot x [Npt x C]
        support_x_low = torch.tensor_split(support_x_low, support_offset_low) # coords
        if support_base_y is not None:
            support_base_y = torch.tensor_split(
                support_base_y, support_offset_low
            )
        
        # split origin support point cloud to transform coords
        support_offset_origin = support_offset[:-1].long().cpu()
        support_x_split = torch.tensor_split(support_x, support_offset_origin)

        support_feat_wobkffn_split = torch.tensor_split(support_feat_wobkffn, support_offset_low)

        # split support_y_low to calculate prototypes
        support_y_low_split = torch.tensor_split(support_y_low, support_offset_low)
        
        support_mask_proposals = []


        # if not self.training:
        #     support_proposals = support_proposals[0] + support_proposals[1]
        # if support_proposals is not None:
        #     for i, proposals in enumerate(support_proposals):
        #         sup_pro = proposals.cuda(non_blocking=True)
        #         if sup_pro.shape[0] == 0:
        #             sup_pro = torch.zeros(size=(1, support_x_split[i].shape[0]), device= support_x_split[i].device)
        #         m2_dict = {tuple(row.tolist()): idx for idx, row in enumerate(support_x_split[i][:,:3])}
        #         indices_list = [m2_dict[tuple(mask.tolist())] for mask in support_x_low[i]]
        #         indices_to_keep_tensor = torch.tensor(indices_list, dtype=torch.long).cuda(non_blocking=True)
        #         sup_pro = sup_pro[:, indices_to_keep_tensor]
        #         support_mask_proposals.append(sup_pro)

        # get prototypes
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset_low)
        bg_mask = torch.tensor_split(bg_mask, support_offset_low)

        #print(self.afha1)
        # For k_shot, extract N_pt/k_shot per shot
        fg_prototypes = self.getPrototypes(
            support_x_low,
            support_feat,
            fg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C
        bg_prototype = self.getPrototypes(
            support_x_low,
            support_feat,
            bg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C

        # reduce the number of bg_prototypes to n_subprototypes when N_way > 1
        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(
                bg_prototype.permute(1, 0)
            ).permute(1, 0)

        sparse_embeddings = torch.cat(
            [bg_prototype, fg_prototypes]
        )  # (N_way+1)*N_pt, C


        support_global_prototypes = torch.cat([sup_feat[sup_label.bool()].mean(dim = 0, keepdim=True)
                                     for sup_feat, sup_label in zip(support_feat, support_y_low_split)], dim = 0)
    
        # get downsampled query features
        (
            query_feat,  # N_q, C
            query_x_low,  # N_q, 3
            query_offset_low,
            query_y_low,  # N_q
            q_base_pred,  # N_q, N_base_classes
            query_base_y,  # N_q
            query_feat_wobkffn,
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # split query features into list according to offset
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_y_low_split = torch.tensor_split(query_y_low, query_offset_low_cpu)
        query_x_low_list = torch.tensor_split(
            query_x_low, query_offset_low_cpu
        )
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(
                query_base_y, query_offset_low_cpu
            )
        
        # split the origin query feature into list
        query_offest_origin = query_offset[:-1].long().cpu()
        query_x_split = torch.tensor_split(query_x, query_offest_origin)
        query_feat_wobkffn_split = torch.tensor_split(query_feat_wobkffn, query_offset_low_cpu)

        # update base prototypes
        if self.training:
            for base_feat, base_y in zip(
                list(query_feat) + list(support_feat),
                list(query_base_y_list) + list(support_base_y),
            ):
                cur_baseclsses = base_y.unique()
                cur_baseclsses = cur_baseclsses[
                    cur_baseclsses != 0
                ]  # remove background
                for class_label in cur_baseclsses:
                    class_mask = base_y == class_label
                    class_features = (
                        base_feat[class_mask].sum(dim=0) / class_mask.sum()
                    ).detach()  # C prototype
                    # use the current features to update when the base prototype is initialized
                    if torch.all(self.base_prototypes[class_label - 1] == 0):
                        self.base_prototypes[class_label - 1] = class_features
                    else:  # exponential moving average
                        self.base_prototypes[class_label - 1] = (
                            self.base_prototypes[class_label - 1] * 0.995
                            + class_features * 0.005
                        )
            # mask out the base porotype for current target classes which should not be considered as background
            # the target classes for current episode should not be considered as background, so mask them out
            mask_list = [
                self.base_class_to_pred_label[base_cls] - 1
                for base_cls in sampled_classes
            ]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool
            )
            base_mask[mask_list] = False
            base_avail_pts = self.base_prototypes[base_mask]
            assert len(base_avail_pts) == self.base_classes - self.n_way
        else:
            base_avail_pts = self.base_prototypes

        # query_mask_proposals = []

        # if query_proposals is not None:
        #     for i, proposals in enumerate(query_proposals):
        #         qry_pro = proposals.cuda(non_blocking=True)
        #         if qry_pro.shape[0] == 0:
        #             qry_pro = torch.zeros(size=(1, query_x_split[i].shape[0]), device= query_x_split[i].device)
        #         m2_dict = {tuple(row.tolist()): idx for idx, row in enumerate(query_x_split[i][:,:3])}
        #         indices_list = [m2_dict[tuple(mask.tolist())] for mask in query_x_low_list[i]]
        #         indices_to_keep_tensor = torch.tensor(indices_list, dtype=torch.long).cuda(non_blocking=True)
        #         qry_pro = qry_pro[:, indices_to_keep_tensor]
        #         query_mask_proposals.append(qry_pro)


        query_pred = []
        count = torch.tensor(0).cuda()
        for i, q_feat in enumerate(query_feat):
          
            # get base guidance, warm up for 1 epoch

            # query_pesudo, prototypes, pred_logits1, pred_logits2, flag = self.get_mask(
            #     query_mask_proposals[i],
            #     q_feat, support_feat, support_y_low_split, query_y_low_split[i], support_mask_proposals, i
            # )

            # if flag:
            #     count = count + torch.tensor(1).cuda()

            # mb = sparse_embeddings[100:, :].clone() 
            # global_query_for_class1 = prototypes[0]
            # global_query_for_class2 = prototypes[1]
            # seq_len, dim = global_query_for_class1.shape
            # mb1 = mb[:100,:]
            # mb2 = mb[100:, :]
        
            # qry = mb[:100,:].reshape(1, 100, 4, 48).type(torch.half)
            # key = global_query_for_class1.reshape(1, seq_len, 4, dim // 4).type(torch.half)
            # value = global_query_for_class1.reshape(1, seq_len, 4, dim // 4).type(torch.half)
            
            # for _ in range(self.num_clustering):
            #     qry = qry + flash_attn_func(qry, key, value)

            # mb1 = qry.reshape(100, 192).type(torch.float).clone()
            
           
            # seq_len, dim = global_query_for_class2.shape
            
            # qry = mb[100:,:].reshape(1, 100, 4, 48).type(torch.half)
            # key = global_query_for_class2.reshape(1, seq_len, 4, dim // 4).type(torch.half)
            # value = global_query_for_class2.reshape(1, seq_len, 4, dim // 4).type(torch.half)
            
            # for _ in range(self.num_clustering):
            #     qry = qry + flash_attn_func(qry, key, value)

            # mb2 = qry.reshape(100, 192).type(torch.float).clone()
          
            new_sparse_embeddings = sparse_embeddings.clone()
            #new_sparse_embeddings[100:, :] = torch.cat([mb1,mb2], dim = 0)


            if epoch < 1:
                base_guidance = None # reasonable
            else:
                base_similarity = F.cosine_similarity(
                    q_feat[:, None, :],  # N_q, 1, C
                    base_avail_pts[None, :, :],  # 1, N_base_classes, C
                    dim=2,
                )  # N_q, N_base_classes
                # max similarity for each query point as the base guidance
                base_guidance = base_similarity.max(dim=1, keepdim=True)[
                    0
                ]  # N_q, 1

            correlations = F.cosine_similarity(
                q_feat[:, None, :], # N X 1 X C
                sparse_embeddings[None, :, :],  # 1, (N_way+1)*N_pt, C
                dim=2,
            )  # N_q, (N_way+1)*N_pt
           
            correlations = (
                self.lin1(
                    correlations.view(
                        correlations.shape[0], self.n_way + 1, -1
                    )  # N_q, (N_way+1), N_pt,
                )
                .permute(2, 1, 0)
                .unsqueeze(0)
            )  # 1, C, N_way+1, N_q

            for layer in self.agglayers:
                correlations = layer(
                    correlations, base_guidance
                )  # 1, C, N_way+1, N_q

            correlations = (
                correlations.squeeze(0).permute(2, 1, 0).contiguous()
            )  # N_q, N_way+1, C

            # reduce the class dimension
            if self.n_way == 1:
                correlations = self.class_reduce(correlations).squeeze(
                    1
                )  # N_q, C
            else:
                correlations = self.class_reduce(
                    correlations.view(correlations.shape[0], -1)
                )  # N_q, C

            # kpconv layer
            coord = query_x_low_list[i]  # N_q, 3
            batch = torch.zeros(
                correlations.shape[0], dtype=torch.int64, device=coord.device
            )
            sigma = 2.0
            radius = 2.5 * self.args.grid_size * sigma
            neighbors = tp.ball_query(
                radius,
                self.args.max_num_neighbors,
                coord,
                coord,
                mode="partial_dense",
                batch_x=batch,
                batch_y=batch,
            )[
                0
            ]  # N_q, max_num_neighbors

            correlations = self.kpconv(
                correlations, coord, batch, neighbors.clone()
            )  # N_q, C

            # classification layer
            out = self.cls(correlations)  # N_q, n_way+1
            query_pred.append(out)

        query_pred = torch.cat(query_pred)  # N_q, n_way+1
        try:
            assert not torch.any(
                torch.isnan(query_pred)
            ), "torch.any(torch.isnan(query_pred))"
        except AssertionError as e:
            print(sparse_embeddings)
        loss = self.criterion(query_pred, query_y_low)
        if query_base_y is not None:
            loss += self.criterion_base(q_base_pred, query_base_y.cuda())

        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # 1, n_way+1, N_query

        # wandb visualization
        if self.visualization:
            self.vis(
                query_offset,
                query_x,
                query_y,
                support_offset,
                support_x,
                support_y,
                final_pred,
            )
    
        return query_y_low.int(), loss, query_y_low, query_x_low, final_pred, count

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """
        Get the features of one point cloud from backbone network.

        Args:
            ptclouds: Input point clouds with shape (N_pt, 6), where N_pt is the number of points.
            offset: Offset tensor with shape (b), where b is the number of query scenes.
            gt: Ground truth labels. shape (N_pt).
            query_base_y: Optional base class labels for input point cloud. shape (N_pt).

        Returns:
            feat: Features from backbone with shape (N_down, C), where C is the number of channels.
            coord: Point coords. Shape (N_down, 3).
            offset: Offset for each scene. Shape (b).
            gt: Ground truth labels. Shape (N_down).
            base_pred: Base class predictions from backbone. Shape (N_down, N_base_classes).
            query_base_y: Base class labels for input point cloud. Shape (N_down).
        """
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),  # rgb color
        )  # (N_pt, 3), (N_pt, 3)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1] # shot > 1 或者 way > 1才有意义，就是在单独求出每个点云sample了多少个点。
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()  # N_pt 求出每个点是属于哪一个点云的，也就是哪一个shot. 保证在求每个点的邻居点这个操作只在当前的批次中进行。

        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma # 0.1
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (N_pt, max_num_neighbors) # 每一个点的邻居。

        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # N_pt, 6
        # downsample the input point clouds
        feat, coord, offset, gt, base_pred, query_base_y = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )  # (N_down, C_bc) (N_down, 3) (b), (N_down), (N_down, N_base_classes), (N_down)

        feat_encoded = self.bk_ffn(feat)  # N_down, C
        return feat_encoded, coord, offset, gt, base_pred, query_base_y, feat

    def getPrototypes(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            #只有一轮循环，怀疑是否能收敛。
            protos1 = self.getMutiplePrototypes(
                coord_mask, feat_mask, k
            )  # k, C
            # protos2 = self.getMultiplePrototypesDPC(
            #     coord_mask, feat_mask, k // 2
            # )
            prototypes.append(protos1)
            #prototypes.append(protos2)

        prototypes = torch.cat(prototypes)  # n_way*k_shot*k, C
        return prototypes
    
    
    def getMultiplePrototypesDPC(self, coord, feat, num_prototypes):
        """
        Extract k protpotypes using density peak clustering

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.
        Return:
            prototypes: Extracted prototypes. Shape: (num_prototypes, C).
        """
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])
            return feat 
        
        dpc_index = self.densityPeakClustering(
            coord,
            num_prototypes
        ).long()

        num_prototypes = len(dpc_index)
        dpc_seeds = feat[dpc_index]

        distances = torch.linalg.norm(
            feat[:, None, :] - dpc_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes) 其它点到种子点的距离， 特征距离。

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,) 其它点到哪一个种子点的距离最短。

        prototypes = torch.zeros(
            (num_prototypes, self.feat_dim), device="cuda"
        )
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(
                1
            )  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[dpc_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)  # (C,)

        return prototypes


    def densityPeakClustering(self, coord, cluster_num):
        """
        Density Peak Clsutering

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            cluster_num: Number of clusters. Int
        """
        coord = coord.unsqueeze(0) # reshape to meet the need 1, N_pt, 3
        N = coord.shape[1]
        dist = torch.cdist(coord, coord) # ratio_dpc
        ner = math.ceil(N * self.args.ratio_dpc /100)
        dist_nearest, index_nearest = torch.topk(dist, k=ner, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6
        
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(coord.dtype)
        dist_max = dist.flatten(1).max(dim=-1)[0][:, None, None]
        distx, index_parent = (dist * mask + dist_max * (1 - mask)).min(dim=-1)
        # select clustering center according to score
        score = distx * density # 1 x N
        _, index_down = torch.topk(score, k=cluster_num, dim=-1) # 1 X K下标
        return index_down.squeeze(0)

    def getMutiplePrototypes(self, coord, feat, num_prototypes):
        """
        Extract k prototypes using furthest point samplling

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.
        Return:
            prototypes: Extracted prototypes. Shape: (num_prototypes, C).
        """
        # when the number of points is less than the number of prototypes, pad the points with zero features
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])
            return feat   

        # sample k seeds  by Farthest Point Sampling # 只
        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()  # (num_prototypes,)

        # use the k seeds as initial centers and compute the point-to-seed distance
        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]  # (num_prototypes, feat_dim)
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes) 其它点到种子点的距离， 特征距离。

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,) 其它点到哪一个种子点的距离最短。

        # aggregating each cluster to form prototype
        prototypes = torch.zeros(
            (num_prototypes, self.feat_dim), device="cuda"
        )
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(
                1
            )  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)  # (C,)

        return prototypes
    

    def enhance_query_feat(self, 
                           query_mask_proposals,
                           query_feat,
                           ):
        
      
        masked_query_feat = torch.zeros_like(query_feat,
                                             device=query_feat.device) # N_pt x feat_dim
        for mask in query_mask_proposals:
            selected_points = query_feat[mask.bool()] # N x feat_dim
            if selected_points.shape[0] == 0:
                continue
            else:
                prototype = selected_points.mean(dim = 0, keepdim=True) # 
                feature = masked_query_feat[mask.bool()]
                masked_query_feat[mask.bool()] = feature + (prototype)
        
        return masked_query_feat
    

    def get_mask(self, query_proposals, feat_q, feat_s, label_s, label_y, support_proposals, count = None, sample_class = None):

        def compute_inter(m1, m2):
            intersection = m1 * m2
            # 计算 m1 中前景像素的数量
            m1_foreground = m1.sum()
            if m1_foreground == 0:
                return torch.tensor(0).cuda()
            # 计算交集中的前景像素数量
            intersection_foreground = intersection.sum()

            # 计算交集前景占 m1 前景的比例
            # 注意要避免 m1 中没有前景像素的情况，防止除以 0
            epsilon = 1e-8
            foreground_ratio = intersection_foreground / (m1_foreground)

            return foreground_ratio
        
        feat_dim = feat_q.shape[1]
        # first, calculate prototypes for each support_feat and its corresponding label
        sup_proto_list = []
        sup_proto_score_list = []

        support_x_y = []
        support_x_y_score = []

        for i, (sup_feat, sup_label) in enumerate(zip(feat_s, label_s)):
            if torch.sum(sup_label) == 0:
                continue
            tmp_pros = [sup_feat[sup_label.bool()].mean(dim = 0)]
            tmp_scores = [torch.tensor(1, dtype=torch.float32).cuda(non_blocking=True)]
            support_x_y.append(sup_feat[sup_label.bool()].mean(dim = 0, keepdim=True))
            support_x_y_score.append(torch.tensor(1, dtype=torch.float32).cuda(non_blocking=True))

            for proposal in support_proposals[i]:
                if torch.sum(proposal) == 0:
                   continue
                proto = sup_feat[proposal.bool()].mean(dim = 0) # C
                score = compute_inter(proposal.float(), sup_label.float()) # 1
                tmp_pros.append(proto)
                tmp_scores.append(score)
            
            tmp_pros = torch.stack(tmp_pros, dim = 0)
            tmp_scores = torch.stack(tmp_scores)

            indices = torch.nonzero(tmp_scores == 1, as_tuple=True)[0]
            tmp_pros = tmp_pros[indices, :]
            tmp_scores = tmp_scores[indices]

            sup_proto_list.append(tmp_pros)
            sup_proto_score_list.append(tmp_scores)
         
        # Second, calculate query prototypes

        sup_proto_list[0] = torch.cat(sup_proto_list[0:5], dim = 0)
        sup_proto_list[1] = torch.cat(sup_proto_list[5:10], dim = 0)
        sup_proto_score_list[0] = torch.cat(sup_proto_score_list[0:5], dim=0)
        sup_proto_score_list[1] = torch.cat(sup_proto_score_list[5:10], dim = 0)

        # sup_proto_list[0] = torch.mean(sup_proto_list[0], dim = 0, keepdim=True)
        # sup_proto_list[1] = torch.mean(sup_proto_list[1], dim = 0, keepdim=True)
        # sup_proto_score_list[0]= torch.tensor(1).cuda()
        # sup_proto_score_list[1]= torch.tensor(1).cuda()
        support_x_y[0] = torch.cat(support_x_y[0:5], dim = 0)
        support_x_y[0] = torch.mean(support_x_y[0], dim = 0, keepdim=True)
        support_x_y[1] = torch.cat(support_x_y[5:10], dim = 0)
        support_x_y[1] = torch.mean(support_x_y[1], dim = 0, keepdim=True)
        support_x_y_score[0] = torch.tensor(1).cuda()
        support_x_y_score[1] = torch.tensor(1).cuda()

        prototype_q = []
        new_proposals = []
        for proposal in query_proposals:
            if torch.sum(proposal) == 0:
                continue
            proto = feat_q[proposal.bool()].mean(dim = 0) # C
            prototype_q.append(proto)
            new_proposals.append(proposal)

        if len(prototype_q) == 0:
            prototype_q = torch.zeros((1, feat_dim)).cuda()
        else:
            prototype_q = torch.stack(prototype_q, dim = 0)

        # Then, calcuate the correlations between prototypes, and weighted by scores
        query_pesudo_label = torch.zeros(feat_q.shape[0], device=label_y.device)

        #return_prototypes2 = torch.cat([sup_proto_list[0], sup_proto_list[1]], dim = 0)


        #return_prototypes2 = torch.cat(return_prototypes2, dim = 0) # 2 x C

        # response = F.cosine_similarity(feat_q.unsqueeze(1), return_prototypes2.unsqueeze(0), dim = 2)

        # threshold = 0.9

        # # 筛选相似度大于阈值的点
        # high_sim_1 = response[:,0] > threshold  # 类别1相似度高的点
        # high_sim_2 = response[:,1] > threshold  # 类别2相似度高的点

        # # 分别统计点数
        # count_class1 = high_sim_1.sum().item()
        # count_class2 = high_sim_2.sum().item()


        # 分别计算与support和query之间的相似度

        return_prototypes = [torch.zeros(size=(1, feat_dim)).cuda()] * 2
        pred_for_class1 = []
        pred_for_class2 = []
        scor1_max, scor2_max = None, None
        for proto, proposal in zip(prototype_q, new_proposals):
            score = F.cosine_similarity(sup_proto_list[0], proto.unsqueeze(0), dim = 1) # N 
            v = (score * sup_proto_score_list[0])
            score = v.max() * v.mean()
            pred_for_class1.append(score * proposal)

        for proto, proposal in zip(prototype_q, new_proposals):
            score = F.cosine_similarity(sup_proto_list[1], proto.unsqueeze(0), dim = 1) # N 
            v = (score * sup_proto_score_list[1])
            score = v.max() * v.mean()
            pred_for_class2.append(score * proposal)

        
        for proto, proposal in zip(prototype_q, new_proposals):
            score = F.cosine_similarity(support_x_y[0], proto.unsqueeze(0), dim = 1) # N 
            v = (score * support_x_y_score[0])
            score = v.max() * v.mean()
            if scor1_max == None or score > scor1_max:
                scor1_max = score

        for proto, proposal in zip(prototype_q, new_proposals):
            score = F.cosine_similarity(support_x_y[1], proto.unsqueeze(0), dim = 1) # N 
            v = (score * support_x_y_score[1])
            score = v.max()
            if scor2_max == None or score > scor2_max:
                scor2_max = score

        if len(pred_for_class1) == 0:
            return query_pesudo_label.int(), return_prototypes, True, True, True
        
        pred1 = torch.stack(pred_for_class1, dim = 0)
        pred2 = torch.stack(pred_for_class2, dim = 0)

        pred1_logits,_ = torch.max(pred1, dim = 0) # N_q
        pred2_logits,_ = torch.max(pred2, dim = 0) # N_q

        pred1_logits = (pred1_logits - pred1_logits.min()) / (pred1_logits.max() - pred1_logits.min() + 1e-8)
        pred2_logits = (pred2_logits - pred2_logits.min()) / (pred2_logits.max() - pred2_logits.min() + 1e-8)

        pred1_logits[pred1_logits >= 0.9] = 1
        pred1_logits[pred1_logits < 0.9] = 0
        pred2_logits[pred2_logits >= 0.9] = 1
        pred2_logits[pred2_logits < 0.9] = 0

        flag = False


        label_unique = torch.unique(label_y)

        # if count_class1 > count_class2 and torch.any(label_unique == 1):
        #     flag = True
        
        # if count_class2 > count_class1 and torch.any(label_unique == 2):
        #     flag = True

        
        # print("count1", count_class1)
        # print("count2", count_class2)



        if scor1_max > scor2_max and not torch.equal(pred1_logits, torch.zeros_like(pred1_logits).cuda()):
            return_prototypes[0] = feat_q[pred1_logits.bool()].mean(dim = 0, keepdim=True)
            if torch.any(label_unique == 1):
                flag = True

        
        if scor2_max > scor1_max and not torch.equal(pred2_logits, torch.zeros_like(pred2_logits).cuda()):
            return_prototypes[1] = feat_q[pred2_logits.bool()].mean(dim = 0, keepdim=True)
            if torch.any(label_unique == 2):
                flag = True


        label_unique = torch.unique(label_y)

        #print("unique label", label_unique)
        # print("count1", count_class1)
        # print("count2", count_class2)
        # print("socre1", scor1_max)
        # print("score2", scor2_max)
        
        return query_pesudo_label.int(), return_prototypes, pred1_logits, pred2_logits, flag

        
        
    

        
    

        

    def vis(
        self,
        query_offset,
        query_x,
        query_y,
        support_offset,
        support_x,
        support_y,
        final_pred,
    ):
        query_offset_cpu = query_offset[:-1].long().cpu()
        query_x_splits = torch.tensor_split(query_x, query_offset_cpu)
        query_y_splits = torch.tensor_split(query_y, query_offset_cpu)
        vis_pred = torch.tensor_split(final_pred, query_offset_cpu, dim=-1)
        support_offset_cpu = support_offset[:-1].long().cpu()
        vis_mask = torch.tensor_split(support_y, support_offset_cpu)

        sp_nps, sp_fgs = [], []
        for i, support_x_split in enumerate(
            torch.tensor_split(support_x, support_offset_cpu)
        ):
            sp_np = (
                support_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            sp_np[:, 3:6] = sp_np[:, 3:6] * 255.0
            sp_fg = np.concatenate(
                (
                    sp_np[:, :3],
                    vis_mask[i].unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            sp_nps.append(sp_np)
            sp_fgs.append(sp_fg)

        qu_s, qu_gts, qu_pds = [], [], []
        for i, query_x_split in enumerate(query_x_splits):
            qu = (
                query_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            qu[:, 3:6] = qu[:, 3:6] * 255.0
            result_tensor = torch.where(
                query_y_splits[i] == 255,
                torch.tensor(0, device=query_y.device),
                query_y_splits[i],
            )
            qu_gt = np.concatenate(
                (
                    qu[:, :3],
                    result_tensor.unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            q_prd = np.concatenate(
                (
                    qu[:, :3],
                    vis_pred[i]
                    .squeeze(0)
                    .max(0)[1]
                    .unsqueeze(-1)
                    .detach()
                    .cpu()
                    .numpy(),
                ),
                axis=-1,
            )

            qu_s.append(qu)
            qu_gts.append(qu_gt)
            qu_pds.append(q_prd)

        wandb.log(
            {
                "Support": [
                    wandb.Object3D(sp_nps[i]) for i in range(len(sp_nps))
                ],
                "Support_fg": [
                    wandb.Object3D(sp_fgs[i]) for i in range(len(sp_fgs))
                ],
                "Query": [wandb.Object3D(qu_s[i]) for i in range(len(qu_s))],
                "Query_pred": [
                    wandb.Object3D(qu_pds[i]) for i in range(len(qu_pds))
                ],
                "Query_GT": [
                    wandb.Object3D(qu_gts[i]) for i in range(len(qu_gts))
                ],
            }
        )()()
