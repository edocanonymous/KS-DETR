import math
import warnings
from typing import Optional, List

import torch
from torch import Tensor

# interpolate, adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d,
from torch.nn.functional import adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d


# ---------------------------
# change the average pool to max pool for area
# -------------------
def interpolate_modified(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> Tensor:  # noqa: F811
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,
            its length has to match `input.dim()`.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recomputed_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    .. warning::
        When scale_factor is specified, if recompute_scale_factor=True,
        scale_factor is used to compute the output_size which will then
        be used to infer new scales for the interpolation.
        The default behavior for recompute_scale_factor changed to False
        in 1.6.0, and scale_factor is used in the interpolation
        calculation.

    Note:
        {backward_reproducibility_note}
    """
    # if has_torch_function_unary(input):
    #     return handle_torch_function(
    #         interpolate,
    #         (input,),
    #         input,
    #         size=size,
    #         scale_factor=scale_factor,
    #         mode=mode,
    #         align_corners=align_corners,
    #         recompute_scale_factor=recompute_scale_factor,
    #     )

    if mode in ("nearest", "area"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            warnings.warn(
                "Default upsampling behavior when mode={} is changed "
                "to align_corners=False since 0.4.0. Please specify "
                "align_corners=True if the old behavior is desired. "
                "See the documentation of nn.Upsample for details.".format(mode)
            )
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "size shape must match input shape. " "Input is {}D, size is {}".format(dim, len(size))
                )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor is {}".format(dim, len(scale_factor))
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is None:
        # only warn when the scales have floating values since
        # the result for ints is the same with/without recompute_scale_factor
        if scale_factors is not None:
            for scale in scale_factors:
                if math.floor(scale) != scale:
                    warnings.warn(
                        "The default behavior for interpolate/upsample with float scale_factor changed "
                        "in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, "
                        "instead of relying on the computed output size. "
                        "If you wish to restore the old behavior, please set recompute_scale_factor=True. "
                        "See the documentation of nn.Upsample for details. "
                    )
                    break
    elif recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
                for i in range(dim)
            ]
        else:
            assert scale_factors is not None
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]
        scale_factors = None

    if input.dim() == 3 and mode == "nearest":
        return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)

    # if input.dim() == 3 and mode == "area":
    #     assert output_size is not None
    #     return adaptive_avg_pool1d(input, output_size)
    # if input.dim() == 4 and mode == "area":
    #     assert output_size is not None
    #     return adaptive_avg_pool2d(input, output_size)
    # if input.dim() == 5 and mode == "area":
    #     assert output_size is not None
    #     return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_max_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_max_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_max_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return torch._C._nn.upsample_linear1d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        return torch._C._nn.upsample_bilinear2d(input, output_size, align_corners, scale_factors)
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return torch._C._nn.upsample_trilinear3d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        return torch._C._nn.upsample_bicubic2d(input, output_size, align_corners, scale_factors)

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        " (got {}D) for the modes: nearest | linear | bilinear | bicubic | trilinear"
        " (got {})".format(input.dim(), mode)
    )