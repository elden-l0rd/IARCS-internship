# To use Llama 2

1. python3 -m pip install -r requirements.txt
2. python3 convert.py --outfile models/13B/ggml-model-f16.bin --outtype f16 ../../llama2/meta_models/llama-2-13b-chat
3. ./quantize  ./models/13B/ggml-model-f16.bin ./models/13B/ggml-model-q4_0.bin q4_0
4. ./main -m ./models/13B/ggml-model-q4_0.bin -n 1024 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/chat-with-bob.txt