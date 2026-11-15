.PHONY: run install validate build ingest train-plan train eval clean corpus-files corpus-web corpus-build corpus-load corpus-bible review review-apply review-status

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

cloud-plan:     ## validate the dataset for hosted fine-tuning (no GPU, no network)
	python3 finetune/train_openai.py --dry-run

cloud-train:    ## hosted fine-tune: uploads + starts the job (needs OPENAI_API_KEY)
	python3 finetune/train_openai.py

notebook:       ## open the free-T4 Colab notebook path
	@echo "Upload the repo, or run: jupyter lab finetune/colab_qlora.ipynb"

corpus-files:   ## books/handbooks/newspapers in ./books ./handbooks ./newspapers ./tv
	python3 -m corpus.ingest_files books/ handbooks/ newspapers/ tv/ --out files

corpus-web:     ## check one URL against robots.txt before crawling
	@read -p "URL to check: " u; python3 -m corpus.ingest_web --check-robots "$$u"

corpus-build:   ## merge + dedupe + report
	python3 -m corpus.build_corpus

corpus-bible:  ## the Luganda Bible: ./bayibuli (verse-numbered .txt per book)
	python3 -m corpus.ingest_bible bayibuli/ --license "check the edition"

review:         ## build the native-review sheet -> corpus/out/review.html
	python3 -m corpus.make_review_sheet

review-apply:   ## apply verdicts from corpus/out/verdicts.txt
	python3 -m corpus.make_review_sheet --apply corpus/out/verdicts.txt

review-status:  ## how many words are signed off, per source kind
	python3 -m corpus.make_review_sheet --status

corpus-load:    ## push the corpus into the running app's memory
	python3 -m corpus.build_corpus --push-to-app http://localhost:8000

clean:
	rm -rf finetune/out/*.jsonl finetune/out/eval_report.json finetune/out/human_rubric.csv
