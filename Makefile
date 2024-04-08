run_all: conda_initial fix_error_of_package downloading_file
conda_initial:
	conda create -y --name gpt4tool python=3.9
	conda activate gpt4tool
fix_error_of_package:
	cd GPT4Tools
	sed '/basicsr/d' requirements.txt > tmp_requirements.txt 
	mv tmp_requirements.txt requirements.txt
	pip install basicsr --use-pep517
	pip install -r requirements.txt
	pip install gradio pillow
	cd ..
downloading_file:
	cd GPT4Tools
	mkdir cache_dir
	python3 scripts/download.py --model-names "lmsys/vicuna-7b-v1.5" --cache-dir cache_dir  
	wget -O vicuna-7b-v1.5-gpt4tools.pth.tar "https://drive.google.com/uc?export=download&id=1UdA6_iOxXZs2V13adLa_V605Ty19KR4s"
mv vicuna-7b-v1.5-gpt4tools.pth.tar cache_dir/
	cd cache_dir && tar -xvf vicuna-7b-v1.5-gpt4tools.pth.tar
	cd ..
	export CACHE_DIR=cache_dir
	export TRANSFORMERS_CACHE=cache_dir
	export HUGGINGFACE_HUB_CACHE=cache_dir
	python gpt4tools_demo.py --base_model cache_dir/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d --lora_model cache_dir/vicuna-7b-v1.5-gpt4tools --cache-dir cache_dir