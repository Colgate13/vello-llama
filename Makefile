.DEFAULT_GOAL := help
.PHONY: %

# Thin wrapper around the `vello-llama-local` CLI so `make <cmd>` keeps working.
# Run `./vello-llama-local help` for the full reference.

%:
	@./vello-llama-local $(MAKECMDGOALS) $(filter-out $@,$(MAKECMDGOALS))
