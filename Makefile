.PHONY: run install validate build ingest train-plan train eval clean

run:            ## start the chatbot on :8000
	python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

install:        ## web app deps only
	python3 -m pip install -r requirements.txt

validate:       ## check every dataset before spending GPU time
	python3 finetune/validate_dataset.py finetune/data

build:          ## merge + dedupe + split into finetune/out
	python3 finetune/build_dataset.py

ingest:         ## example: pull a public Luganda instruction set
	python3 finetune/ingest_hf.py --dataset allandclive/alpaca_4k_luganda --preset alpaca --limit 4000

train-plan:     ## show the training plan, touch no GPU
	python3 finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml --dry-run

train:          ## the real thing (GPU box)
	python3 finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml

eval:           ## chrF++ / BLEU / LLM judge + native-speaker sheet
	python3 finetune/eval.py --judge

clean:
	rm -rf finetune/out/*.jsonl finetune/out/eval_report.json finetune/out/human_rubric.csv
