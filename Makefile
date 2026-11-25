.PHONY: eval index-full

eval:
	.venv/bin/python -m src.eval.capstone_hard_gates

index-full:
	.venv/bin/python -m src.rag_ingestion.full_index complete
