import torch
import torch.autograd as ag

try:
    from os.path import join as pjoin, dirname
    from torch.utils.cpp_extension import load as load_extension
    root_dir = pjoin(dirname(__file__), 'src_pytorch13')
    _depthconv = load_extension(
        '_depthconv',
        [pjoin(root_dir, 'depthconv_cuda_redo.c'), pjoin(root_dir, 'depthconv_cuda_kernel.cu')],
        verbose=True
    )
    #print(_depthconv)
except ImportError:
    raise ImportError('Can not compile depth-aware cnn library.')


#how to use this file:
#python load_depthconv.py

#if depthconv is compiled successfully, then test (in the folder of the whole project)
#python -c 'from ltr.external.depthconv.modules import DepthConv; a=DepthConv()'
