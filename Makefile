PY := .venv/bin/python
JUPYTEXT := .venv/bin/jupytext
JUPYTER := .venv/bin/jupyter
MMDC := npx -y @mermaid-js/mermaid-cli@11

.PHONY: test diagrams dashboard pipeline

test:
	$(PY) -m pytest -q

dashboard:
	.venv/bin/streamlit run app/dashboard.py

NOTEBOOKS := 02_sourcing 03_preprocessing_eda 05_training 06_model_selection 07_validation \
	08_deployment 09_monitoring 10_recommendations

pipeline: $(addprefix nb-,$(NOTEBOOKS))

nb-%:
	$(JUPYTEXT) --to ipynb notebooks/$*.py
	$(JUPYTER) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/$*.ipynb

diagrams:
	@for f in reports/diagrams/*.mmd; do \
		out=reports/figures/$$(basename $$f .mmd).png; \
		$(MMDC) -i $$f -o $$out -s 4 -b white || exit 1; \
	done
