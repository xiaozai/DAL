import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math



class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, feature_extractor_depth=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.feature_extractor_depth = feature_extractor_depth

        self.conv_0 = nn.Conv2d(int(2*self.filter_initializer.feature_dim), self.filter_initializer.feature_dim, kernel_size=3, padding=3 // 2)



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
        # Init weights
        for m in self.feature_extractor_depth.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_feat_depth, test_feat_depth, train_bb, *args, **kwargs):
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
        if train_feat_depth.dim() == 5:
            train_feat_depth = train_feat_depth.view(-1, *train_feat_depth.shape[-3:])
        if test_feat_depth.dim() == 5:
            test_feat_depth = test_feat_depth.view(-1, *test_feat_depth.shape[-3:])

        # print(['train_feat', train_feat.shape])
        # print(['train_feat_depth', train_feat_depth.shape])
        # ['train_feat', torch.Size([15, 1024, 18, 18])]
        # ['train_feat_depth', torch.Size([15, 1024, 18, 18])]


        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)
        train_feat_depth = self.extract_classification_feat_depth(train_feat_depth, num_sequences)
        test_feat_depth  = self.extract_classification_feat_depth(test_feat_depth, num_sequences)

        # print(['train_feat', train_feat.shape])
        # print(['train_feat_depth', train_feat.shape])
        # ['train_feat', torch.Size([3, 6, 512, 18, 18])]
        # ['train_feat_depth', torch.Size([3, 6, 512, 18, 18])]
        n_img_in_seq, n_seq=train_feat.shape[0], train_feat.shape[1]
        train_feat= torch.cat((train_feat, train_feat_depth), dim=2)
        test_feat = torch.cat((test_feat, test_feat_depth), dim=2)
        train_feat =train_feat.view(-1, train_feat.shape[-3], train_feat.shape[-2], train_feat.shape[-1])
        test_feat =test_feat.view(-1, test_feat.shape[-3], test_feat.shape[-2], test_feat.shape[-1])
        train_feat= self.conv_0(train_feat)
        test_feat = self.conv_0(test_feat)
        train_feat =train_feat.view(n_img_in_seq, n_seq, train_feat.shape[-3], train_feat.shape[-2], train_feat.shape[-1])
        test_feat =test_feat.view(n_img_in_seq, n_seq, test_feat.shape[-3], test_feat.shape[-2], test_feat.shape[-1])

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

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

    def extract_classification_feat_depth(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor_depth is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor_depth(feat)

        output = self.feature_extractor_depth(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
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

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses
