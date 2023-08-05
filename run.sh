# echo "quant+compiled_individually.log"
# time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --profile "quant+compiled individually" --dynamic_quant 2



echo "quant+compiled_normally.log"
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --profile "dq+comp" --dynamic_quant 1

echo "quant+compiled max optimize.log"
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --profile "dq+comp+maxopt" --dynamic_quant 1 --max_optimize true

echo "compiled.log"
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --profile "comp"

echo "quant+no_compile.log"
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile false --profile "dq" --dynamic_quant 1

echo "compiled.log max-optimize"
time python generate.py --prompt "Hello, my name is" --max_new_tokens 200 --num_samples 10 --fake false  --compile true --profile "comp+maxopt" --max_optimize true
