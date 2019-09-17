#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from copy import deepcopy
from utilities.nd_softmax import softmax_helper
import numpy as np
from network_architecture.initialization import InitWeights_He
from network_architecture.neural_network import SegmentationNetwork
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.unet_model import build_preprocessor
from modeling.aspp import *
# cenet
# from modeling.CENet import *

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = nn.LeakyReLU(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs_first_conv,
                                     self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                     self.nonlin, self.nonlin_kwargs)] +
              [ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, self.conv_op,
                                     self.conv_kwargs,
                                     self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                     self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm1d):
        print(str(module), module.training)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (224, 224)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 32
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000 # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        # self.convolutional_upsampling = convolutional_upsampling
        # self.convolutional_pooling = convolutional_pooling
        # self.upscale_logits = upscale_logits
        # if nonlin_kwargs is None:
        #      nonlin_kwargs = {'negative_slope':1e-2, 'inplace':True}
        # if dropout_op_kwargs is None:
        #     dropout_op_kwargs = {'p':0.5, 'inplace':True}
        # if norm_op_kwargs is None:
        #     norm_op_kwargs = {'eps':1e-5, 'affine':True, 'momentum':0.1}
        #
        # self.conv_kwargs = {'stride':1, 'dilation':1, 'bias':True}
        #
        # self.nonlin = nonlin
        # self.nonlin_kwargs = nonlin_kwargs
        # self.dropout_op_kwargs = dropout_op_kwargs
        # self.norm_op_kwargs = norm_op_kwargs
        # self.weightInitializer = weightInitializer
        # self.conv_op = conv_op
        # self.norm_op = norm_op
        # self.dropout_op = dropout_op
        # self.num_classes = num_classes
        # self.final_nonlin = final_nonlin
        # self.do_ds = deep_supervision
        #
        # if conv_op == nn.Conv2d:
        #     upsample_mode = 'bilinear'
        #     pool_op = nn.MaxPool2d
        #     transpconv = nn.ConvTranspose2d
        #     if pool_op_kernel_sizes is None:
        #         pool_op_kernel_sizes = [(2, 2)] * num_pool
        #     if conv_kernel_sizes is None:
        #         conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        # elif conv_op == nn.Conv3d:
        #     upsample_mode = 'trilinear'
        #     pool_op = nn.MaxPool3d
        #     transpconv = nn.ConvTranspose3d
        #     if pool_op_kernel_sizes is None:
        #         pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
        #     if conv_kernel_sizes is None:
        #         conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        # else:
        #     raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))
        #
        # self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0)
        # self.pool_op_kernel_sizes = pool_op_kernel_sizes
        # self.conv_kernel_sizes = conv_kernel_sizes
        #
        # self.conv_pad_sizes = []
        # for krnl in self.conv_kernel_sizes:
        #     self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        #
        # self.conv_blocks_context = []
        # self.conv_blocks_localization = []
        # self.td = []
        # self.tu = []
        # self.seg_outputs = []
        #
        # output_features = base_num_features
        # input_features = input_channels
        #
        # for d in range(num_pool):
        #     # determine the first stride
        #     if d != 0 and self.convolutional_pooling:
        #         first_stride = pool_op_kernel_sizes[d-1]
        #     else:
        #         first_stride = None
        #
        #     self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
        #     self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
        #     # add convolutions
        #     self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
        #                                                       self.conv_op, self.conv_kwargs, self.norm_op,
        #                                                       self.norm_op_kwargs, self.dropout_op,
        #                                                       self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
        #                                                       first_stride))
        #     if not self.convolutional_pooling:
        #         self.td.append(pool_op(pool_op_kernel_sizes[d]))
        #     input_features = output_features
        #     output_features = int(np.round(output_features * feat_map_mul_on_downscale))
        #     if self.conv_op == nn.Conv3d:
        #         output_features = min(output_features, self.MAX_NUM_FILTERS_3D)
        #     else:
        #         output_features = min(output_features, self.MAX_FILTERS_2D)
        #
        # # now the bottleneck.
        # # determine the first stride
        # if self.convolutional_pooling:
        #     first_stride = pool_op_kernel_sizes[-1]
        # else:
        #     first_stride = None
        #
        # # the output of the last conv must match the number of features from the skip connection if we are not using
        # # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # # done by the transposed conv
        # if self.convolutional_upsampling:
        #     final_num_features = output_features
        # else:
        #     final_num_features = self.conv_blocks_context[-1].output_channels
        #
        # self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        # self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        # self.conv_blocks_context.append(nn.Sequential(
        #     StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
        #                       self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
        #                       self.nonlin_kwargs, first_stride),
        #     StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
        #                       self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
        #                       self.nonlin_kwargs)))
        #
        # # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        # if not dropout_in_localization:
        #     old_dropout_p = self.dropout_op_kwargs['p']
        #     self.dropout_op_kwargs['p'] = 0.0
        #
        # # now lets build the localization pathway
        # for u in range(num_pool):
        #     nfeatures_from_down = final_num_features
        #     nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
        #     n_features_after_tu_and_concat = nfeatures_from_skip * 2
        #
        #     # the first conv reduces the number of features to match those of skip
        #     # the following convs work on that number of features
        #     # if not convolutional upsampling then the final conv reduces the num of features again
        #     if u != num_pool - 1 and not self.convolutional_upsampling:
        #         final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
        #     else:
        #         final_num_features = nfeatures_from_skip
        #
        #     if not self.convolutional_upsampling:
        #         self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))
        #     else:
        #         self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u+1)],
        #                                   pool_op_kernel_sizes[-(u+1)], bias=False))
        #
        #     self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u+1)]
        #     self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u+1)]
        #     self.conv_blocks_localization.append(nn.Sequential(
        #         StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
        #                           self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
        #                           self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs),
        #         StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
        #                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
        #                           self.nonlin, self.nonlin_kwargs)
        #     ))
        #
        # for ds in range(len(self.conv_blocks_localization)):
        #     self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
        #                                     1, 1, 0, 1, 1, False))
        #
        # self.upscale_logits_ops = []
        # cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        # for usl in range(num_pool - 1):
        #     if self.upscale_logits:
        #         self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl+1]]),
        #                                                 mode=upsample_mode))
        #     else:
        #         self.upscale_logits_ops.append(lambda x: x)
        #
        # if not dropout_in_localization:
        #     self.dropout_op_kwargs['p'] = old_dropout_p
        #
        # # register all modules properly
        # self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        # self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        # self.td = nn.ModuleList(self.td)
        # self.tu = nn.ModuleList(self.tu)
        # self.seg_outputs = nn.ModuleList(self.seg_outputs)
        # if self.upscale_logits:
        #     self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops) # lambda x:x is not a Module so we need to distinguish here
        #
        # if self.weightInitializer is not None:
        #     self.apply(self.weightInitializer)
        #     #self.apply(print_module_training_status)

        # dawn changed here
        # the model generator is changed in a very rude way
        self.conv_op = conv_op
        self.num_classes = num_classes
        # deeplab

        BatchNorm = SynchronizedBatchNorm2d
        input_channels = 3
        self.preprocessor = build_preprocessor(input_channels, 3, BatchNorm)
        self.backbone = build_backbone('resnet', 16, BatchNorm)
        self.aspp = build_aspp('resnet', 16, BatchNorm)
        self.decoder = build_decoder(num_classes, 'resnet', BatchNorm)


        # CENET
        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        #
        # self.dblock = DACblock(512)
        # self.spp = SPPblock(512)


        # self.decoder4 = DecoderBlock(516, filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        #
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # nnunet
        # skips = []
        # seg_outputs = []
        # # np.savez('/data2/dawn/test', images=x)
        # for d in range(len(self.conv_blocks_context) - 1):
        #     x = self.conv_blocks_context[d](x)
        #     skips.append(x)
        #     # print(x.size())
        #     if not self.convolutional_pooling:
        #         x = self.td[d](x)
        #
        # x = self.conv_blocks_context[-1](x)
        #
        # for u in range(len(self.tu)):
        #     x = self.tu[u](x)
        #     # print(x.size())
        #     x = torch.cat((x, skips[-(u + 1)]), dim=1)
        #     x = self.conv_blocks_localization[u](x)
        #     seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
        #
        # if self.do_ds:
        #     return tuple([seg_outputs[-1]] + [i(j) for i, j in
        #                                       zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        # else:
        #     return seg_outputs[-1]

        # deeplab

        x = self.preprocessor(x)
        y, low_level_feat = self.backbone(x)
        y = self.aspp(y)
        y = self.decoder(y, low_level_feat)
        # pytorch 0.4.0
        y = F.upsample(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        # 0.4.1
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)


        # CENet
        # x = self.firstconv(x)
        # x = self.firstbn(x)
        # x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        # e1 = self.encoder1(x)
        # e2 = self.encoder2(e1)
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)
        #
        # # Center
        # e4 = self.dblock(e4)
        # e4 = self.spp(e4)
        # Decoder
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        #
        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        # y = self.finalconv3(out)

        return y

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """

        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64(5 * np.prod(map_size) * base_num_features + num_modalities * np.prod(map_size) + \
              num_classes * np.prod(map_size))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = 5 if p < (npool -1) else 2 # 2 + 2 for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size) * num_feat
            # print(p, map_size, num_feat, tmp)
        return tmp


