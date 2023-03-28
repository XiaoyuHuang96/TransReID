import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # print(x.shape, y.shape)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # print(dist_mat[is_pos].contiguous().shape)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_example_mining_FCD(dist_mat, ref_dist_mat, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, M]
      ref_dist_mat: pytorch Variable, pair wise distance between samples, shape [N, M]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == ref_dist_mat.size(0)
    assert dist_mat.size(1) == ref_dist_mat.size(1)
    N, M = dist_mat.size(0), dist_mat.size(1)

    # shape [N, 1]
    dist_an = [] #torch.ones(N)

    for i in range(N):
      # print(dist_mat[i].dtype)
      # print(ref_dist_mat[i].dtype)
      mat_tmp = dist_mat[i].clone()
      # [N, M]
      mat_tmp1 = mat_tmp.expand(N, M)
      # [N]
      delta_mat = torch.sqrt(torch.sum(torch.pow(mat_tmp1 - ref_dist_mat, 2), dim=1))
      delta_mat[i] = 0.0
      sorted, indices = torch.sort(delta_mat)
      dist_an.append(sorted[1])

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    # dist_ap = dist_ap.squeeze(1)
    
    dist_an = torch.Tensor(dist_an).unsqueeze(1)
    # print("dist_an",dist_an.size())

    # if return_inds:
    #     # shape [N, N]
    #     ind = (labels.new().resize_as_(labels)
    #            .copy_(torch.arange(0, N).long())
    #            .unsqueeze(0).expand(N, N))
    #     # shape [N, 1]
    #     p_inds = torch.gather(
    #         ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    #     n_inds = torch.gather(
    #         ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    #     # shape [N]
    #     p_inds = p_inds.squeeze(1)
    #     n_inds = n_inds.squeeze(1)
    #     return dist_ap, dist_an, p_inds, n_inds

    return dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        # print("dist_mat", dist_mat.shape)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class RSDLoss(object):
    """
    Relation similarity distillation loss using similarity between two features
    """
    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor


    def __call__(self, global_feat, ref_global_feat, normalize_feature=False):
        # print(global_feat.shape, ref_global_feat.shape)
        # print(global_feat, ref_global_feat)
        dist_mat = torch.matmul(global_feat, global_feat.t())
        ref_dist_mat = torch.matmul(ref_global_feat, ref_global_feat.t())
        if normalize_feature:
            dist_mat = normalize(dist_mat, axis=-1)
            ref_dist_mat = normalize(ref_dist_mat, axis=-1)
        # dist_mat = cosine_dist(global_feat, global_feat)
        # print("dist_mat", dist_mat.shape)
        # ref_dist_mat = cosine_dist(ref_global_feat, ref_global_feat)
        # print("ref_dist_mat", dist_mat.shape)
        N = dist_mat.size(0)
        loss = torch.pow(dist_mat - ref_dist_mat, 2).sum() / (N * N)

        return loss

class SimLoss(object):
    """
    Sim loss using cosine similarity between two features
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        # if margin is not None:
        #     self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        # else:
        #     self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, ref_global_feat, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            ref_global_feat = normalize(ref_global_feat, axis=-1)
        dist_mat = cosine_dist(global_feat, global_feat)
        # print("dist_mat", dist_mat.shape)
        ref_dist_mat = cosine_dist(ref_global_feat, ref_global_feat)
        # print("ref_dist_mat", dist_mat.shape)
        N = dist_mat.size(0)
        loss = ((dist_mat - ref_dist_mat) * (dist_mat - ref_dist_mat)).sum() / (N * N)
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        # dist_ap *= (1.0 + self.hard_factor)
        # dist_an *= (1.0 - self.hard_factor)

        # y = dist_an.new().resize_as_(dist_an).fill_(1)
        # if self.margin is not None:
        #     loss = self.ranking_loss(dist_an, dist_ap, y)
        # else:
        #     loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss



class FCDistLoss(object):
    """
    Feature Consistent distillation loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=0.3, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, ref_global_feat, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            ref_global_feat = normalize(ref_global_feat, axis=-1)
        # dist_mat = euclidean_dist(global_feat, global_feat)
        dist_an = hard_example_mining_FCD(global_feat, ref_global_feat)
        
        N = global_feat.size(0)
        M = global_feat.size(1)
        # print(N, global_feat.size(), ref_global_feat.shape)
        # print(global_feat.device, ref_global_feat.device, dist_an.device)
        loss = self.margin + torch.sqrt(torch.pow(global_feat-ref_global_feat, 2)) - dist_an.cuda()
        tmp = torch.zeros(N,M).cuda()
        loss = torch.where(loss < 0, tmp, loss)
        loss = loss.sum() / N
        # dist_ap *= (1.0 + self.hard_factor)
        # dist_an *= (1.0 - self.hard_factor)

        # y = dist_an.new().resize_as_(dist_an).fill_(1)
        # if self.margin is not None:
        #     loss = self.ranking_loss(dist_an, dist_ap, y)
        # else:
        #     loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_an