ROM   ?= pokeyellow.gb
TASK  ?= yellow_pikachu
STATE ?= states/yellow_lab.state
OUT   ?= episodes/poke.h5
PY    ?= uv run python

.PHONY: help sync test state smoke route calibrate baselines collect collect-big train probe finetune plan eval sweep clean

help:
	@echo "sync        install deps from pyproject"
	@echo "test        run the suite (no ROM, no GPU needed)"
	@echo "state       play in a window, close it to save $(STATE)"
	@echo "smoke       detected game + RAM readings + one A press"
	@echo "route       record the demonstration by hand (one keypress = one action)"
	@echo "calibrate   confirm this ROM's party_count address"
	@echo "baselines   every constant-button policy + uniform random"
	@echo "collect     small dataset -> $(OUT)"
	@echo "collect-big 800 x 48 random + 200 noisy -> $(OUT)"
	@echo "train       10 epochs, ~64 min on an M5 via MPS"
	@echo "probe       linear probe on the frozen encoder"
	@echo "finetune    rollout fine-tuning (encoder frozen)"
	@echo "plan/eval   one CEM plan, then execute it"
	@echo "sweep       100 planner seeds + cost-vs-outcome AUC"
	@echo ""
	@echo "override with e.g. make collect TASK=leave_room OUT=episodes/test.h5"

sync:
	uv sync

test:
	uv run pytest -q

state:
	$(PY) -m scripts.capture_state $(ROM) --out $(STATE)

smoke:
	$(PY) -m scripts.smoke_test --rom $(ROM) --task $(TASK) --state $(STATE)

route:
	$(PY) -m scripts.record_route --rom $(ROM) --task $(TASK) --state $(STATE)

calibrate:
	$(PY) -m scripts.calibrate_ram --rom $(ROM) --task $(TASK) --state $(STATE)

baselines:
	$(PY) -m scripts.eval_baselines --rom $(ROM) --task $(TASK) --state $(STATE)

collect:
	$(PY) -m scripts.collect --rom $(ROM) --task $(TASK) --state $(STATE) --out $(OUT) \
		--noisy 32 --random 50 --random-steps 24

collect-big:
	$(PY) -m scripts.collect --rom $(ROM) --task $(TASK) --state $(STATE) --out $(OUT) \
		--random 800 --random-steps 48 --noisy 200

train:
	$(PY) -m scripts.train --data-path $(OUT)

probe:
	$(PY) -m scripts.probe checkpoints/best.pt --data $(OUT)

finetune:
	$(PY) -m scripts.finetune_rollout checkpoints/best.pt --data $(OUT)

plan:
	$(PY) -m scripts.plan_cem checkpoints/rollout/best.pt --data $(OUT) --out plans/$(TASK).json

eval:
	$(PY) -m scripts.eval_plan --plan plans/$(TASK).json --rom $(ROM) --state $(STATE)

sweep:
	$(PY) -m scripts.sweep_seeds checkpoints/rollout/best.pt --rom $(ROM) --task $(TASK) \
		--state $(STATE) --data $(OUT) --seeds 100

clean:
	rm -f before.png after.png frame.png
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
