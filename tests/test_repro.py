import torch

# That won't work because PT2 doesn't support bitcast
# which changes shape https://github.com/pytorch/pytorch/pull/102920
def test_bitcast_view():
    def run_bitcast_view(x: torch.Tensor):
        return x.view(torch.uint8)
    x = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
    y1 = run_bitcast_view(x)
    y2 = torch.compile(run_bitcast_view)(x)
    assert y1 == y2
