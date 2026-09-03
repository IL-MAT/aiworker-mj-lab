.PHONY: help
help:
	@echo "make build|serve|deploy"

.PHONY: build
build:
	@echo "Building MkDocs documentation..."
	cd .. && mkdocs build

.PHONY: serve
serve:
	@echo "Serving MkDocs documentation locally on http://127.0.0.1:8000 ..."
	cd .. && mkdocs serve

.PHONY: deploy
deploy:
	@echo "Deploying documentation to gh-pages..."
	cd .. && mkdocs gh-deploy --force
