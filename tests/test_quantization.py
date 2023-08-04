import torch

from model import quantize_uint8, dequantize_uint8

def run_test_quantization():

    x = torch.rand(10, 8, dtype=torch.bfloat16)
    x_q, coefs = quantize_uint8(x)

    assert x_q.dtype == torch.uint8
    assert x_q.shape == x.shape
    x_deq = dequantize_uint8(x_q, coefs)

    torch.testing.assert_close(x, x_deq, atol=0.05, rtol=0.01, msg=f"{x=}\n{x_deq=}\n{x_deq/x=}")


def test_quantization(compile=True):
    run_test_quantization()
    if compile:
        run_test_quantization_comp = torch.compile(run_test_quantization)
        run_test_quantization_comp()
