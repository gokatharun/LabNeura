PYTHON ?= python3
COMPONENT ?= all
TARGET ?= native
BUILD_TYPE ?= Release
JOBS ?=
MSG ?=

DEV := $(PYTHON) scripts/dev.py --component $(COMPONENT) --target $(TARGET) --build-type $(BUILD_TYPE) $(if $(JOBS),--jobs $(JOBS),)

.PHONY: compile test benchmark coverage check format clean push help

compile:
	$(DEV) compile

test:
	$(DEV) test

benchmark:
	$(DEV) benchmark

coverage:
	$(DEV) coverage

check:
	$(DEV) check

format:
	$(DEV) format

clean:
	$(DEV) clean

push:
	$(PYTHON) scripts/git_push.py --message "$(MSG)" --component $(COMPONENT) --target $(TARGET) $(if $(JOBS),--jobs $(JOBS),)

help:
	@$(PYTHON) scripts/dev.py --help
