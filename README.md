To download weights, follow `download_weights.md`. I used OpenLLaMA weights.

To run benchmark.

Model definition in `model.py`, generation code in `generate.py`.

```
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true
```

```
Current benchmark results
7B, 6 prompt/200 tokens, bf16
eager:
Time for inference 2: 9.29 sec total, 21.53 tokens/sec
Bandwidth achieved: 290.09 GB/s

torch.compile per block
Time for inference 2: 5.03 sec total, 39.74 tokens/sec
Bandwidth achieved: 535.52 GB/s

Note: Still seems primarily overhead-bound (also, cudagraphs is segfaulting for me right now...)

- torch.compile on whole model generates gibberish right now... My suspicion is that it's because we're not guarding on attributes of modules. Yes indeed, it's now fixed)

- cudagraphs now doesn't work because I needed to add in-place mutation to fix whole graph compilation issues

torch.compile on whole model
Time for inference 3: 2.66 sec total, 75.13 tokens/sec
Bandwidth achieved: 1012.55 GB/s

Refactored to also compile sampling code (and also updated pytorch)
Time for inference 3: 1.16 sec total, 86.02 tokens/sec
Bandwidth achieved: 1159.24 GB/s

After enabling cudagraphs, changing kv-cache to be a scatter instead of full copy, and some optimizations in Inductor (such as reinplacing index_put)

Time for inference 5: 0.94 sec total, 106.79 tokens/sec
Bandwidth achieved: 1439.13 GB/s
```

Note: Running on an A100 80GB, albeit power-limited to 330 watts. Empirically, seems like peak bandwidth is about 1700 GB/s.

WIP: int8 KV-cache quantization

With quantization and no compilation the performance is very low
```
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile false --quantize true

Time for inference 2: 78.11 sec total, 2.56 tokens/sec
Bandwidth achieved: 34.51 GB/s
```

Compilation improves that by a lot. With quantization and compilation:
```
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --quantize true

Time for inference 10: 2.72 sec total, 73.43 tokens/sec
Bandwidth achieved: 989.59 GB/s
Memory used: 14.01 GB
```

For comparison, without quantization, but with compilation:
```
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --quantize false

Time for inference 10: 2.48 sec total, 80.78 tokens/sec
Bandwidth achieved: 1088.72 GB/s
Memory used: 13.84 GB
```

It is not yet clear why quantization increases, and not decreases, memory usage
