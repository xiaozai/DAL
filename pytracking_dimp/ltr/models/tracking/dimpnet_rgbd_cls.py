import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.linear_filter_rgbd as target_clf_rgbd
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor


class DiMPnet_rgbd_cls(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        bb_regressor_depth:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression.
        train_feature_extractor:  Whether feature extractor should be trained or not."""

    def __init__(self, feature_extractor, feature_extractor_depth, classifier, bb_regressor,classification_layer, bb_regressor_layer, train_feature_extractor=True):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor_depth=feature_extractor_depth
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

        if not train_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
            for p in self.feature_extractor_depth.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            train_depths: train depth samples (images, sequences, 1, H, W).
            test_depths:  test depth samples (images, sequences, 1, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        #print([train_imgs.shape, train_imgs.view(-1, *train_imgs.shape[-3:]).shape])
        #[torch.Size([3, 10, 4, 288, 288]), torch.Size([30, 4, 288, 288])]
        # train_depths=train_imgs[:,:,3,:,:].clone().view(*train_imgs.shape[:2],1,*train_imgs.shape[3:]).repeat(1,1,3,1,1)
        # test_depths =test_imgs[:,:,3,:,:].clone().view(*test_imgs.shape[:2],1,*test_imgs.shape[3:]).repeat(1,1,3,1,1)
        #print([train_depths.shape, test_depths.shape])
        train_rgbs, train_depths  =train_imgs[:,:,:3,:,:],train_imgs[:,:,3,:,:]
        test_rgbs,  test_depths   =test_imgs[:,:,:3,:,:] ,test_imgs[:,:,3,:,:]
        train_depths = train_depths.view(*train_imgs.shape[:2],1,*train_imgs.shape[3:]).expand(-1,-1,3,-1,-1)
        test_depths  = test_depths.view(*test_imgs.shape[:2],1,*test_imgs.shape[3:]).expand(-1,-1,3,-1,-1)

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_rgbs.view(-1, *train_rgbs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_rgbs.view(-1, *test_rgbs.shape[-3:]))
        # Extract backbone features
        train_feat_depth = self.extract_backbone_features_depth(train_depths.view(-1, *train_depths.shape[-3:]))
        test_feat_depth= self.extract_backbone_features_depth(test_depths.view(-1, *test_depths.shape[-3:]))


        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)
        train_feat_clf_depth = self.get_backbone_clf_feat(train_feat_depth)
        test_feat_clf_depth = self.get_backbone_clf_feat(test_feat_depth)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)
        train_feat_depth_iou = self.get_backbone_bbreg_feat(train_feat_depth)
        test_feat_depth_iou = self.get_backbone_bbreg_feat(test_feat_depth)

        #print([train_feat_clf.shape, test_feat_clf.shape])[torch.Size([18, 1024, 18, 18]), torch.Size([18, 1024, 18, 18])]


        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_feat_clf_depth, test_feat_clf_depth,train_bb, *args, **kwargs)

        train_feat_iou[0]= torch.cat((train_feat_iou[0], train_feat_depth_iou[0]), dim=1)
        train_feat_iou[1]= torch.cat((train_feat_iou[1], train_feat_depth_iou[1]), dim=1)

        test_feat_iou[0]= torch.cat((test_feat_iou[0], test_feat_depth_iou[0]), dim=1)
        test_feat_iou[1]= torch.cat((test_feat_iou[1], test_feat_depth_iou[1]), dim=1)

        # for feat in train_feat_iou:
        #     print(feat.shape)
        # torch.Size([18, 1024, 36, 36])
        # torch.Size([18, 2048, 18, 18])

        # train_feat_iou=train_feat_iou + train_feat_depth_iou
        # test_feat_iou =test_feat_iou  + test_feat_depth_iou
        # for feat in train_feat_iou:
        #     print(feat.shape)
        # torch.Size([18, 512, 36, 36])
        # torch.Size([18, 1024, 18, 18])
        # torch.Size([18, 512, 36, 36])
        # torch.Size([18, 1024, 18, 18])


        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_classification_feat_depth(self, backbone_feat):
        return self.classifier.extract_classification_feat_depth(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_backbone_features_depth(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor_depth(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



@model_constructor
def dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf')):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)
    clf_feature_extractor_d = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)


    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear_rgbd(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf_rgbd.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer,
                                         feature_extractor=clf_feature_extractor,
                                         feature_extractor_depth=clf_feature_extractor_d)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet_rgbd(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet_rgbd_cls(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf')):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    backbone_net_depth = backbones.resnet50(pretrained=backbone_pretrained)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)
    clf_feature_extractor_d = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf_rgbd.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer,
                                         feature_extractor=clf_feature_extractor,
                                         feature_extractor_depth=clf_feature_extractor_d)
    # Bounding box regressor for rgb
    bb_regressor = bbmodels.AtomIoUNet_rgbd(input_dim=(8*128,8*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)


    # DiMP network
    net = DiMPnet_rgbd_cls(feature_extractor=backbone_net, feature_extractor_depth=backbone_net_depth,classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
