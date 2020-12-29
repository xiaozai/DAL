import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math
import torch.nn.functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.layers.blocks import conv_block


class FilterPool(nn.Module):
    """Pool the target region in a feature map.
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(filter_size, filter_size, 1/feature_stride)
        self.pool_square = pool_square

    def forward(self, feat, bb):
        """Pool the regions in bb.
        args:
            feat:  Input feature maps. Dims (num_samples, feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (num_samples, 4).
        returns:
            pooled_feat:  Pooled features. Dims (num_samples, feat_dim, wH, wW)."""

        # Add batch_index to rois
        bb = bb.view(-1,4)
        num_images_total = bb.shape[0]
        batch_index = torch.arange(num_images_total, dtype=torch.float32).view(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        pool_bb = bb.clone()

        if self.pool_square:
            bb_sz = pool_bb[:, 2:4].prod(dim=1, keepdim=True).sqrt()
            pool_bb[:, :2] += pool_bb[:, 2:]/2 - bb_sz/2
            pool_bb[:, 2:] = bb_sz

        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        roi1 = torch.cat((batch_index, pool_bb), dim=1)

        return self.prroi_pool(feat, roi1)

class LogisticFilter(nn.Module):
    """occ classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features.

        """

    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(2048, 1024, kernel_size=3, padding=3 // 2)
        self.conv_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=3 // 2)
        self.conv_2 = nn.Conv2d(512, 128, kernel_size=3, padding=3 // 2)
        self.conv_3 = nn.Conv2d(128, 4, kernel_size=3, padding=3 // 2)

        self.linear0= nn.Linear(1386, 512, bias=True)
        self.bn0 = nn.BatchNorm1d(512)
        self.relu0 = nn.ReLU(inplace=True)

        self.linear1= nn.Linear(512, 64, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2= nn.Linear(64, 1, bias=True)
        self.softmax= nn.Softmax(dim=2)
        self.sigmoid= nn.Sigmoid()
        #self.filter_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=3 // 2)
        self.filter_pool = FilterPool(filter_size=1, feature_stride=16, pool_square=False)

        # self.filter_size = filter_size
        # # Modules
        # self.filter_initializer = filter_initializer
        # self.filter_optimizer = filter_optimizer
        # self.feature_extractor = feature_extractor
        #
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat_clf, test_feat_clf, train_feat_clf_depth, test_feat_clf_depth, target_score, iou_pred, train_bb, proposals2, *args, **kwargs):
        """Learns a occlusion classification filter based on all inputs provided.
        args:
            # train_feat:  Backbone features for the train samples (4 or 5 dims).
            # test_feat:  Backbone features for the test samples (4 or 5 dims).
            # trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            train_feat_clf: torch.Size([12, 1024, 18, 18])
            test_feat_clf: torch.Size([12, 1024, 18, 18])
            train_feat_clf_depth: torch.Size([12, 1024, 18, 18])
            test_feat_clf_depth: torch.Size([12, 1024, 18, 18])
            target_score: , [torch.Size([3, 4, 19, 19])]
            iou_pred: torch.Size([3, 4, 8])
            train_bb: torch.Size([3, 4, 4])]
            proposals2: torch.Size([3, 4, 8, 4])]
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            occ_state:  Classification scores on the test samples."""

        n_img_in_seq, n_seq=iou_pred.shape[0], iou_pred.shape[1]
        n_proposal = proposals2.shape[2]
        proposals2 = proposals2.view(n_img_in_seq*n_seq, -1, 4) #(12,8,4)

        train_bb=train_bb.view(n_img_in_seq*n_seq,4)

        ref_feat=torch.cat((train_feat_clf, train_feat_clf_depth), dim=1)
        test_feat =torch.cat((test_feat_clf, test_feat_clf_depth), dim=1)
        #print([ref_feat.shape, test_feat.shape])[torch.Size([12, 2048, 18, 18]), torch.Size([12, 2048, 18, 18])]
        ref_feat=self.conv_0(ref_feat)
        test_feat=self.conv_0(test_feat)
        #print([ref_feat.shape, test_feat.shape])[torch.Size([12, 1024, 18, 18]), torch.Size([12, 1024, 18, 18])]
        ref_feat = self.conv_1(ref_feat)
        test_feat= self.conv_1(test_feat)

        ref_feat = self.filter_pool(ref_feat, train_bb) #torch.Size([12, 512, 1, 1])
        test_feat= self.filter_pool(test_feat, proposals2) #torch.Size([96, 512, 1, 1])

        ref_feat = ref_feat.view(n_img_in_seq, n_seq, 1, -1) #(3, 4, 1, 512)
        ref_feat = ref_feat.repeat(1,1,n_proposal,1) #(3,4,8,512)
        test_feat= test_feat.view(n_img_in_seq,n_seq, n_proposal, -1) #(3, 4, 8, 512)
        target_score=target_score.view(n_img_in_seq, n_seq, 1, -1).repeat(1,1,n_proposal,1) #(3,4,1,19*19) -> (3,4,8,361)
        iou_pred = iou_pred.view(n_img_in_seq,n_seq,-1,1) #(3,4,8,1)
        feat = torch.cat((ref_feat,test_feat, target_score, iou_pred), dim=3) #torch.Size([3, 4, 8, 1386])

        # feat=self.relu0(self.bn0(self.linear0(feat).view(-1,512)).view(n_img_in_seq,n_seq,n_proposal,-1))
        # feat=self.relu1(self.bn1(self.linear1(feat).view(-1,64)).view(n_img_in_seq,n_seq,n_proposal,-1))
        feat=self.relu0(self.linear0(feat))
        feat=self.relu1(self.linear1(feat))
        feat=self.linear2(feat)
        occ_state=self.sigmoid(feat) #torch.Size([3, 4, 8, 1])
        return occ_state
