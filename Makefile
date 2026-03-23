TARGET_DIR = ./fits
RANDOM_SEED = 2
YEARS = 2020 2010 2000 1990 1980 1970

PYTHON = uv run python
DATA_DIR = tennis_atp
CORES = 2
CHAINS = 2

# Methods that take (year, target_dir, seed)
SIMPLE_METHODS = pymc cmdstanpy nutpie_pymc nutpie_stan

# NumPyro: (year, platform, chain_method, target_dir, seed)
NUMPYRO_CONFIGS = gpu_parallel cpu_parallel
# NUMPYRO_CONFIGS = gpu_parallel gpu_vectorized cpu_parallel cpu_vectorized

# BlackJAX: (year, platform, target_dir, seed, chain_method)
BLACKJAX_CONFIGS = gpu_parallel cpu_parallel
# BLACKJAX_CONFIGS = cpu_parallel gpu_vectorized

# Build all target file lists
SIMPLE_TARGETS = $(foreach m,$(SIMPLE_METHODS),$(foreach y,$(YEARS),$(TARGET_DIR)/$(m)/samples_$(y).netcdf))
NUMPYRO_TARGETS = $(foreach c,$(NUMPYRO_CONFIGS),$(foreach y,$(YEARS),$(TARGET_DIR)/pymc_numpyro_$(c)/samples_$(y).netcdf))
BLACKJAX_TARGETS = $(foreach c,$(BLACKJAX_CONFIGS),$(foreach y,$(YEARS),$(TARGET_DIR)/pymc_blackjax_$(c)/samples_$(y).netcdf))

ALL_TARGETS = $(SIMPLE_TARGETS) $(NUMPYRO_TARGETS) $(BLACKJAX_TARGETS)

.PHONY: all clean

all: $(ALL_TARGETS)

$(DATA_DIR):
	git clone --depth 1 https://github.com/JeffSackmann/tennis_atp.git $@

# Simple methods: pymc, cmdstanpy, nutpie_pymc, nutpie_stan
define simple_rule
$(TARGET_DIR)/$(1)/samples_$(2).netcdf: fit_$(1).py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting $(1) $(2)"
	@mkdir -p $(TARGET_DIR)/$(1)
	$(PYTHON) fit_$(1).py $(2) $(TARGET_DIR) $(RANDOM_SEED) $(CORES) $(CHAINS)
endef

$(foreach m,$(SIMPLE_METHODS),$(foreach y,$(YEARS),$(eval $(call simple_rule,$(m),$(y)))))

# NumPyro: split config into platform and chain_method
define numpyro_rule
$(TARGET_DIR)/pymc_numpyro_$(1)_$(2)/samples_$(3).netcdf: fit_pymc_numpyro.py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting pymc_numpyro $(1) $(2) $(3)"
	@mkdir -p $(TARGET_DIR)/pymc_numpyro_$(1)_$(2)
	$(PYTHON) fit_pymc_numpyro.py $(3) $(1) $(2) $(TARGET_DIR) $(RANDOM_SEED) $(CORES) $(CHAINS)
endef

$(foreach p,gpu cpu,$(foreach y,$(YEARS),$(eval $(call numpyro_rule,$(p),parallel,$(y)))))

# BlackJAX: split config into platform and chain_method
define blackjax_rule
$(TARGET_DIR)/pymc_blackjax_$(1)_$(2)/samples_$(3).netcdf: fit_pymc_blackjax.py fetch_data.py sackmann.py | $(DATA_DIR)
	@echo "Fitting pymc_blackjax $(1) $(2) $(3)"
	@mkdir -p $(TARGET_DIR)/pymc_blackjax_$(1)_$(2)
	$(PYTHON) fit_pymc_blackjax.py $(3) $(1) $(TARGET_DIR) $(RANDOM_SEED) $(2) $(CORES) $(CHAINS)
endef

$(foreach p,gpu cpu,$(foreach y,$(YEARS),$(eval $(call blackjax_rule,$(p),parallel,$(y)))))

clean:
	rm -rf $(TARGET_DIR)
