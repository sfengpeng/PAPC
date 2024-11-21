import torch


def forward_mask_func(args, sam, qry_feat, sup_feat, sup_offest = None, qry_offest = None):
    if args.n_way == 1:
        if args.k_shot == 5:
            return forward_1way5shot(sam, qry_feat, sup_feat, sup_offest)
        elif args.k_shot == 1:
            return forward_1way1shot(sam, qry_feat, sup_feat)
    elif args.n_way == 2:
        if args.k_shot == 1:
            return forward_2way1shot(sam, qry_feat, sup_feat, sup_offest, qry_offest)
        elif args.k_shot == 5:
            return forward_2way5shot(sam, qry_feat, sup_feat, sup_offest, qry_offest)

def forward_1way1shot(sam, qry_feat, sup_feat):
    pass



def forward_1way5shot(sam, qry_feat, sup_feat, sup_offest):
    """
    qry_feat: N_q x 6
    sup_feat: List [N_si x 6]
    sup_offest: support_offest
    """
    support_offset = sup_offest[:-1].long().cpu()
    support_feat = torch.tensor_split(sup_feat, support_offset)

    # obtain support proposals
    support_proposals = []
    for feat in support_feat:
        feat_num = feat.shape[0]

        support_proposal,_ = sam.forward_mask_proposals(
                    feature = feat[:, 3:6].contiguous(),
                    coord = feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
        if support_proposal is None:
            support_proposal = torch.zeros(size=(1, feat_num))
        support_proposals.append(support_proposal)
    
    # obtain query proposals
    
    query_proposals, _ = sam.forward_mask_proposals(
                    feature = qry_feat[:, 3:6].contiguous(),
                    coord = qry_feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
    if query_proposals is None:
        query_proposals = torch.zeros(size=(1, qry_feat.shape[0]))
    
    return support_proposals, query_proposals



def forward_2way1shot(sam, qry_feat, sup_feat, sup_offest = None, qry_offest = None):
    support_offset = sup_offest[:-1].long().cpu()
    support_feat = torch.tensor_split(sup_feat, support_offset)

    query_offest = qry_offest[:-1].long().cpu()
    query_feat = torch.tensor_split(qry_feat, query_offest)

    support_proposals = []
    for feat in support_feat:
        feat_num = feat.shape[0]

        support_proposal,_ = sam.forward_mask_proposals(
                    feature = feat[:, 3:6].contiguous(),
                    coord = feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
        if support_proposal is None:
            support_proposal = torch.zeros(size=(1, feat_num))
        support_proposals.append(support_proposal)
    
    query_proposals = []
    for feat in query_feat:
        feat_num = feat.shape[0]

        query_proposal,_ = sam.forward_mask_proposals(
                    feature = feat[:, 3:6].contiguous(),
                    coord = feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
        if query_proposal is None:
            query_proposal = torch.zeros(size=(1, feat_num))
        query_proposals.append(query_proposal)
    
    return support_proposals, query_proposals
    

def forward_2way5shot(sam, qry_feat, sup_feat, sup_offest = None, qry_offest = None):
    support_offset = sup_offest[:-1].long().cpu()
    support_feat = torch.tensor_split(sup_feat, support_offset)

    query_offest = qry_offest[:-1].long().cpu()
    query_feat = torch.tensor_split(qry_feat, query_offest)

    support_proposals = []
    for feat in support_feat:
        feat_num = feat.shape[0]

        support_proposal,_ = sam.forward_mask_proposals(
                    feature = feat[:, 3:6].contiguous(),
                    coord = feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
        if support_proposal is None:
            support_proposal = torch.zeros(size=(1, feat_num))

        torch.cuda.empty_cache()
        support_proposals.append(support_proposal)

    query_proposals = []
    for feat in query_feat:
        feat_num = feat.shape[0]

        query_proposal,_ = sam.forward_mask_proposals(
                    feature = feat[:, 3:6].contiguous(),
                    coord = feat[:, :3].contiguous(),
                    num_seeds = 1024,
                    flag = True,
            )
        if query_proposal is None:
            query_proposal = torch.zeros(size=(1, feat_num))
        torch.cuda.empty_cache()
        query_proposals.append(query_proposal)

    
    return support_proposals, query_proposals