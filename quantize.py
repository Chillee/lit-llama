import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch._dynamo import is_compiling as dynamo_is_compiling

def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    r"""
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.

    Note: no error checking for torch.compiled path, if input.shape = [i, j] and j<=16 then the triton kernel
    will error.

    Args:
        input (Tensor, int8): the first tensor to be multiplied
        mat2 (Tensor, int8): the second tensor to be multiplied

    Return:
        out (Tensor, int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if dynamo_is_compiling():
        return torch._int_mm(input, mat2)

    # error checking for cublas path
    assert (
        mat2.device == input.device
    ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
    device_cpu = "cpu" in [mat2.device.type, input.device.type]
    # with input.shape = [i,j] and mat2.shape = [j,k]
    i_is_strictly_greater_than_16 = input.shape[0] > 16
    j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
    k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
    bad_dimensions_for_cublas = not (
        i_is_strictly_greater_than_16
        and j_is_nonzero_multiple_of_8
        and k_is_nonzero_multiple_of_8
    )

    # fallback path
    if device_cpu or bad_dimensions_for_cublas:
        if input.shape[0]==1 or mat2.shape[-1]==1:
            return (input.to(torch.int32).unsqueeze(2) * mat2.to(torch.int32).unsqueeze(0)).sum(dim=1)

        return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
            input.device.type
        )

    # cublas paths
    if not mat2.is_contiguous():  # silently gives incorrect result without this
        mat2 = mat2.contiguous()
    if (not input.is_contiguous()) and (
        input.shape[0] % 8 != 0
    ):  # gives cryptic error without this
        input.contiguous() # (it seems the transpose makes cublas check the above j constraint on i)
    return torch._int_mm(input, mat2)

#keep
# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    scales = t.abs().max(dim=-1, keepdim=True)[0].float() # want float scales to avoid overflows
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales

# keep
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point

#keep
def quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    if not dynamo_is_compiling():
        mm_out = torch.matmul(x, w_vals_int8_t.to(x.dtype))*w_scales
    else:
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
        mm_out = quant_int8_per_token_matmul(
            x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype)
    if bias is not None:
        mm_out += bias
    return mm_out

#keep
def quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert x_vals_int8.dtype == torch.int8, \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8_t.dtype == torch.int8, \
        f'w dtype {w_vals_int8_t.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, \
        f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)

    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    assert x_scales.dtype == torch.float, f"x_scales needs to be a torch.float32 but got {x_scales.dtype}"

    y = y_dot_int32 * x_scales * w_scales
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y


class DynamicallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`, implementing dynamic quantization on
    the input across all axes except for the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer.

        This method applies dynamic quantization to the input tensor across all axes except
        the last axis using the `quant_int8_dynamic_per_token_linear` function.

        Args:
            X (torch.Tensor): The input tensor to the quantized linear layer.

        Returns:
            torch.Tensor: The output tensor after the quantized matmul and rescale.

        """
        Y = quant_int8_dynamic_per_token_linear(
            X, self.W_int_repr_t, self.W_scales, self.bias, X.dtype)
        return Y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear) -> 'DynamicallyPerAxisQuantizedLinear':
        """
        Converts a `mod` of class `torch.nn.Linear` to the dynamically quantized version of it.

        Note: this class does not require calibration.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            DynamicallyPerAxisQuantizedLinear: The converted quantized linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -128, 127, torch.int8)
        new_mod.register_buffer('W_int_repr_t', W_int_repr.contiguous().t())
        new_mod.W_scales = nn.Parameter(W_scales)
        new_mod.bias = mod.bias
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=''
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == '':
            new_fqn = name
        else:
            new_fqn = f'{cur_fqn}.{name}'
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn)

def apply_dynamic_quant(model):
    replace_with_custom_fn_if_matches_filter(
        model,
        DynamicallyPerAxisQuantizedLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))
