import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math
import torch.nn.functional as F


class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, settings, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.settings = settings
        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
    #     """Learns a target classification filter based on the train samples and return the resulting classification
    #     scores on the test samples.
    #     The forward function is ONLY used for training. Call the individual functions during tracking.
    #     args:
    #         train_feat:  Backbone features for the train samples (4 or 5 dims).
    #         test_feat:  Backbone features for the test samples (4 or 5 dims).
    #         trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
    #         *args, **kwargs:  These are passed to the optimizer module.
    #     returns:
    #         test_scores:  Classification scores on the test samples."""
    #
    #     assert train_bb.dim() == 3
    #
    #     num_sequences = train_bb.shape[1]
    #
    #     if train_feat.dim() == 5:
    #         train_feat = train_feat.view(-1, *train_feat.shape[-3:])
    #     if test_feat.dim() == 5:
    #         test_feat = test_feat.view(-1, *test_feat.shape[-3:])
    #
    #     # Extract features
    #     train_feat = self.extract_classification_feat(train_feat, num_sequences)
    #     test_feat = self.extract_classification_feat(test_feat, num_sequences)
    #
    #     # Train filter
    #     filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)
    #     # Classify samples using all return filters
    #     test_scores = [self.classify(f, test_feat) for f in filter_iter]
    #     return test_scores

    def forward(self, train_feat, test_feat, train_bb,train_depths, test_depths, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]
        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)
        print('Song in ltr.models.target_classifier.line_filter.py Line 91, before get_filter ..')
        # Train filter
        #if self.settings.
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, train_depths, test_depths, *args, **kwargs)


        if self.settings.depthaware_for_classiferonline:
            print('Song in ltr.models.target_classifier.line_filter.py Line 98, before depthaware_classify')
            test_depths=F.upsample(test_depths, size=(test_feat.shape[3],test_feat.shape[4]), mode='bilinear')
            test_depths=test_depths.view(test_depths.shape[0], test_depths.shape[1], 1, test_depths.shape[2], test_depths.shape[3])#([3, 4, 1, 18, 18])
            filter=filter.view(1, filter.shape[0], filter.shape[1], filter.shape[2], filter.shape[3])
            filter_iter=[f.view(1, f.shape[0], f.shape[1], f.shape[2], f.shape[3]) for f in filter_iter]#([1, 4, 512, 4, 4]) for each
            test_scores = [self.depthaware_classify(f, test_feat, test_depths, self.settings.depthaware_alpha) for f in filter_iter] # Song !!!!! alpha in Eq.3

        else:
            # Classify samples using all return filters
            test_scores = [self.classify(f, test_feat) for f in filter_iter]


        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def depthaware_classify(self, weights, feat, depth, alpha):
        """Run classifier (filter) on the features (feat)."""
        print('Song ltr.models.target_classifier.line_filter.py Line 131, in depthaware_classify before filter_layer.applyfilter_depthware ...')
        if feat.shape[2]!=depth.shape[2] or feat.shape[2]!=depth.shape[2]:
            depth=F.upsample(depth, size=(feat.shape[2],feat.shape[3]),mode='bilinear')
        #print('depthaware_classify',feat.shape, depth.shape)
        scores = filter_layer.apply_filter_depthaware(feat, weights,depth, alpha)

        return scores

    def get_filter(self, feat, bb, train_depths, test_depths, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb, train_depths, test_depths)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, train_depths=train_depths, test_depths=test_depths, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None
        #self.target_mask=self.filter_optimizer.target_mask

        return weights, weights_iter, losses
