TARGET_DIR = ./fits
RANDOM_SEED = 2
YEARS = 2020 2019 2015 2010 2000 1990 1980 1968

PYTHON = uv run python
DATA_DIR = tennis_atp
CORES = 16

# Methods that take (year, target_dir, seed)
SIMPLE_METHODS = pymc cmdstanpy nutpie_pymc nutpie_stan

# NumPyro: (year, platform, chain_method, target_dir, seed)
NUMPYRO_CONFIGS = gpu_parallel gpu_vectorized cpu_parallel cpu_vectorized

# BlackJAX: (year, platform, target_dir, seed, chain_method)
BLACKJAX_CONFIGS = cpu_parallel gpu_vectorized

# Build all target file lists
SIMPLE_TARGETS = $(foreach m,$(SIMPLE_METHODS),$(foreach y,$(YEARS),$(TARGET_DIR)/$(m)/runtime_$(y).txt))
NUMPYRO_TARGETS = $(foreach c,$(NUMPYRO_CONFIGS),$(foreach y,$(YEARS),$(TARGET_DIR)/pymc_numpyro_$(c)/runtime_$(y).txt))
BLACKJAX_TARGETS = $(foreach c,$(BLACKJAX_CONFIGS),$(foreach y,$(YEARS),$(TARGET_DIR)/pymc_blackjax_$(c)/runtime_$(y).txt))

ALL_TARGETS = $(SIMPLE_TARGETS) $(NUMPYRO_TARGETS) $(BLACKJAX_TARGETS)

REPORT = Compare runtimes.html

.PHONY: all clean

all: $(REPORT)

$(REPORT): Compare\ runtimes.qmd $(ALL_TARGETS)
	@mkdir -p plots
	uv run quarto render "Compare runtimes.qmd"

$(DATA_DIR):
	git clone --depth 1 https://github.com/JeffSackmann/tennis_atp.git $@

# Simple methods: pymc, cmdstanpy, nutpie_pymc, nutpie_stan
define simple_rule
$(TARGET_DIR)/$(1)/runtime_%.txt: fit_$(1).py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting $(1) $$*"
	@mkdir -p $(TARGET_DIR)/$(1)
	$(PYTHON) fit_$(1).py $$* $(TARGET_DIR) $(RANDOM_SEED)
endef

$(foreach m,$(SIMPLE_METHODS),$(eval $(call simple_rule,$(m))))

# NumPyro: split config into platform and chain_method
define numpyro_rule
$(TARGET_DIR)/pymc_numpyro_$(1)_$(2)/runtime_%.txt: fit_pymc_numpyro.py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting pymc_numpyro $(1) $(2) $$*"
	@mkdir -p $(TARGET_DIR)/pymc_numpyro_$(1)_$(2)
	$(PYTHON) fit_pymc_numpyro.py $$* $(1) $(2) $(TARGET_DIR) $(RANDOM_SEED)
endef

$(foreach p,gpu cpu,$(foreach c,parallel vectorized,$(eval $(call numpyro_rule,$(p),$(c)))))

# BlackJAX: split config into platform and chain_method
define blackjax_rule
$(TARGET_DIR)/pymc_blackjax_$(1)_$(2)/runtime_%.txt: fit_pymc_blackjax.py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting pymc_blackjax $(1) $(2) $$*"
	@mkdir -p $(TARGET_DIR)/pymc_blackjax_$(1)_$(2)
	$(PYTHON) fit_pymc_blackjax.py $$* $(1) $(TARGET_DIR) $(RANDOM_SEED) $(2)
endef

$(foreach p,gpu cpu,$(foreach c,parallel vectorized,$(eval $(call blackjax_rule,$(p),$(c)))))

clean:
	rm -rf $(TARGET_DIR) plots "Compare runtimes.html" "Compare runtimes_files"
