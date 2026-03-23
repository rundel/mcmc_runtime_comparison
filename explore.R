library(ncdf4)
library(tidyverse)

read_runtime <- function(path) {
  nc <- nc_open(path)
  on.exit(nc_close(nc))

  vars <- names(nc$var)

  if ("sample_stats/perf_counter_diff" %in% vars) {
    perf_diff    <- ncvar_get(nc, "sample_stats/perf_counter_diff")
    process_diff <- ncvar_get(nc, "sample_stats/process_time_diff")
    tibble(
      file    = path,
      chain   = seq_len(ncol(perf_diff)),
      perf    = colSums(perf_diff),
      process = colSums(process_diff)
    )
  } else if ("sample_stats/sampling_time" %in% vars) {
    sampling_time <- ncvar_get(nc, "sample_stats/sampling_time")
    tibble(
      file    = path,
      chain   = seq_len(ncol(sampling_time)),
      perf    = colSums(sampling_time),
      process = NA_real_
    )
  }
  # else {
  #  # Fallback: read total runtime from txt file, divide equally across chains
  #  txt_path <- str_replace(path, "samples_(\\d+)\\.netcdf", "runtime_\\1.txt")
  #  total    <- as.numeric(readLines(txt_path))
  #  lp_var   <- intersect(vars, c("sample_stats/lp", "sample_stats/logp"))[1]
  #  n_chains <- ncvar_get(nc, lp_var) |> ncol()
  #  tibble(
  #    file    = path,
  #    chain   = seq_len(n_chains),
  #    perf    = total / n_chains,
  #    process = NA_real_
  #  )
  #}
}

get_n_games <- function(path) {
  nc <- nc_open(path)
  on.exit(nc_close(nc))
  tryCatch(
    suppressMessages(length(ncvar_get(nc, "observed_data/win_lik"))),
    error = function(e) NA_integer_
  )
}

files <- list.files("fits", pattern = "\\.netcdf$", recursive = TRUE, full.names = TRUE) |>
  keep(\(f) !str_detect(f, "/other/"))

games_by_year <- tibble(file = files) |>
  mutate(year = str_extract(file, "[0-9]{4}") |> as.integer()) |>
  filter(str_detect(file, "/pymc/")) |>
  mutate(n_games = map_int(file, get_n_games)) |>
  select(year, n_games)

runtimes <- map(files, read_runtime, .progress = TRUE) |>
  list_rbind() |>
  mutate(
    framework = str_extract(file, "fits/([^/]+)/", group = 1),
    year      = str_extract(file, "[0-9]{4}") |> as.integer()
  ) |>
  left_join(games_by_year, by = "year") |>
  select(framework, year, chain, n_games, perf, process)

runtimes

framework_labels <- c(
  pymc                      = "PyMC",
  cmdstanpy                 = "Stan",
  nutpie_pymc               = "nutpie (PyMC)",
  nutpie_stan               = "nutpie (Stan)",
  pymc_numpyro_cpu_parallel = "NumPyro (CPU)",
  pymc_numpyro_gpu_parallel = "NumPyro (GPU)",
  pymc_blackjax_cpu_parallel = "BlackJAX (CPU)",
  pymc_blackjax_gpu_parallel = "BlackJAX (GPU)"
)

runtimes <- runtimes |>
  mutate(framework = recode(framework, !!!framework_labels))

runtimes <- runtimes |>
  mutate(iters_per_sec = 1000 / perf)

runtimes_avg <- runtimes |>
  summarise(iters_per_sec = mean(iters_per_sec), .by = c(framework, n_games))

runtimes_end <- runtimes_avg |>
  slice_max(n_games, by = framework)

x_max <- max(runtimes_avg$n_games)

g = ggplot(runtimes_avg, aes(x = n_games, y = iters_per_sec, color = framework, group = framework)) +
  geom_line(linewidth = 1) +
  geom_point(data = runtimes, aes(y = iters_per_sec), alpha = 0.4, size = 2) +
  ggrepel::geom_text_repel(
    data = runtimes_end, aes(x = x_max, label = framework),
    hjust = 0, direction = "y", segment.color = NA,
    xlim = c(x_max * 1.01, Inf)
  ) +
  coord_cartesian(clip = "off") +
  labs(x = "Number of matches", y = "Iterations / second") +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.margin = margin(5.5, 120, 5.5, 5.5)
  )

dir.create("plots", showWarnings = FALSE)

ggsave("plots/runtime_linear.png", g, width = 10, height = 6)
ggsave("plots/runtime_log.png",    g + scale_y_log10(), width = 10, height = 6)
