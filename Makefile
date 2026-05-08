.DEFAULT_GOAL := help
.PHONY: install build-vello vello-clean help

# Convenience wrappers. For everything else, use ./vello directly.

help:
	@./vello-installer help

install:
	@./vello-installer install

build-vello:
	@command -v cargo >/dev/null 2>&1 || . "$$HOME/.cargo/env"; \
	  cd vello-cli && cargo build --release
	@ln -sf vello-cli/target/release/vello vello
	@echo "built ./vello"

vello-clean:
	@command -v cargo >/dev/null 2>&1 || . "$$HOME/.cargo/env"; \
	  cd vello-cli && cargo clean
	@rm -f vello
