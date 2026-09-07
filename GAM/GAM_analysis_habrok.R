# =============================================================================
# GAM_analysis_habrok.R
#
# The whole GAM analysis in one batch script: model fitting, sensitivity fits,
# diagnostics, the linear mixed-model comparison and every figure. Supersedes
# GAM_script_combined.R (fitting) + GAM_local_analysis.R (comparison, figures),
# which had to be run in two places against a model file passed between them.
#
# The valence x progression interaction ti(par_valence, percentage_index,
# by = source) IS the research question -- does the paragraph -> comment
# valence coupling change over the story? It is RETAINED in every model. Its
# significance and concurvity are reported; no AIC or parsimony gate is ever
# allowed to delete it.
#
#   Part A  nested build-up (ML, compareML), descriptive only
#   Part B  refinement (fREML, discrete): AR(1), book structure, weighting
#   Part C  sensitivity fits
#   Part D  diagnostics
#   Part E  linear mixed-model comparison (lme4)
#   Part F  figures
#
# UPDATED FOR THE 2026-09-06 DATA PREPARATION
# -------------------------------------------
# * Input moved to september-data/ and the row count changed: the story table
#   is now the WebNovel paragraph count of the verified segmentation rather
#   than the union of two positional indexes.
# * `zh_text_missing` is gone. The scrape-failure sensitivity now keys on
#   `zh_missing` (no Chinese for that paragraph) and `zh_status`.
# * NEW and important: `comm_anchor`. Qidian comments are anchored to
#   paragraphs through a recovered segmentId -> paragraph mapping. It is exact
#   for the 58 books that keep the publisher's U+3000 indent (comm_anchor = 1)
#   and a proportional approximation for the 49 glued ones (comm_anchor = 0,
#   median error 0 paragraphs, 84% within one). Part C refits on
#   comm_anchor == 1 alone; if the ti() terms hold, the approximation is not
#   driving the result and one sentence in the paper covers it.
# * The prediction grid no longer hard-codes the book-smooth labels to
#   exclude. They are read off the fitted model, so this does not silently
#   break when a different book structure wins the AIC comparison.
#
# Usage (Habrok):  sbatch GAM/gam_job.sh
# Usage (local):   Rscript GAM/GAM_analysis_habrok.R
# Paths can be overridden with the env vars GAM_DATA, GAM_OUT, GAM_FIG.
# =============================================================================

suppressPackageStartupMessages({
  library(mgcv)
  library(itsadug)
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(lme4)
  library(performance)
  library(patchwork)
})

t_start <- Sys.time()

# ---- 0. Paths and data ------------------------------------------------------

first_existing <- function(...) {
  for (p in c(...)) if (nzchar(p) && file.exists(p)) return(p)
  NULL
}

DATA <- Sys.getenv("GAM_DATA")
if (!nzchar(DATA)) {
  DATA <- first_existing(
    "september-data/long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv",
    "../september-data/long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv",
    # the file copied next to the script, as the Habrok job used to expect
    "long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv"
  )
}
if (is.null(DATA) || !file.exists(DATA)) {
  stop("long table not found. Set GAM_DATA to ",
       "september-data/long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv")
}

OUT <- Sys.getenv("GAM_OUT", unset = if (dir.exists("GAM")) "GAM" else ".")
FIG <- Sys.getenv("GAM_FIG", unset = if (dir.exists("figures")) "figures" else ".")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)
op <- function(f) file.path(OUT, f)
fp <- function(f) file.path(FIG, f)

n_threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "8"))
cat("data    :", DATA, "\nmodels  :", OUT, "\nfigures :", FIG,
    "\nthreads :", n_threads, "\n\n")

long_df_roll <- read_csv(DATA, show_col_types = FALSE)

stopifnot(!anyDuplicated(long_df_roll[, c("bookId", "source", "new_index")]))
# Long format invariant: exactly two rows (Qidian, WebNovel) per paragraph.
n_par <- nrow(distinct(long_df_roll, bookId, new_index))
stopifnot(nrow(long_df_roll) == 2 * n_par)
cat("long table rows:", nrow(long_df_roll), "=  2 x", n_par, "paragraphs\n")

required <- c("bookId", "source", "new_index", "par_valence",
              "avg_comm_valence", "comm_count", "percentage_index")
missing <- setdiff(required, names(long_df_roll))
if (length(missing)) stop("input is missing: ", paste(missing, collapse = ", "))

has_anchor <- "comm_anchor" %in% names(long_df_roll)
has_zh_missing <- "zh_missing" %in% names(long_df_roll)

long_df_roll$bookId <- as.factor(long_df_roll$bookId)
long_df_roll$source <- as.factor(long_df_roll$source)

model_df <- long_df_roll %>%
  arrange(bookId, new_index) %>%
  filter(!is.na(avg_comm_valence), !is.na(par_valence), !is.na(percentage_index))
model_df$start_event <- !duplicated(model_df$bookId)

cat("modelled rows:", nrow(model_df), "of", nrow(long_df_roll), "\n")
print(count(model_df, source, name = "rows"))
if (has_anchor) {
  cat("\ncomment anchor (1 = exact segmentId->paragraph link, 0 = proportional):\n")
  print(model_df %>% group_by(source) %>%
          summarise(rows = n(), exact = sum(comm_anchor == 1, na.rm = TRUE),
                    pct_exact = sprintf("%.1f%%", 100 * mean(comm_anchor == 1, na.rm = TRUE)),
                    .groups = "drop"))
}

# =============================================================================
# Part A: nested progression (ML) -- descriptive
# =============================================================================

cat("\n\n########## Part A: nested build-up (ML) ##########\n")

fit_ml <- function(formula, name) {
  m <- bam(formula, data = model_df, method = "ML", nthreads = n_threads)
  cat("\n---", name, "---\n"); print(summary(m))
  saveRDS(m, op(paste0(name, ".rds")))
  m
}

mod0 <- fit_ml(avg_comm_valence ~ par_valence, "mod0")
mod1 <- fit_ml(avg_comm_valence ~ s(par_valence, k = 20), "mod1")
mod2 <- fit_ml(avg_comm_valence ~ s(par_valence, k = 20) + s(percentage_index),
               "mod2")
mod3 <- fit_ml(avg_comm_valence ~ s(par_valence, k = 20) + s(percentage_index)
               + ti(par_valence, percentage_index), "mod3")
mod4 <- fit_ml(avg_comm_valence ~ source + s(par_valence, k = 20)
               + s(percentage_index) + ti(par_valence, percentage_index), "mod4")
mod5 <- fit_ml(avg_comm_valence ~ source + s(par_valence, by = source, k = 20)
               + s(percentage_index, by = source)
               + ti(par_valence, percentage_index, by = source), "mod5")
mod6 <- fit_ml(avg_comm_valence ~ source + s(par_valence, by = source, k = 20)
               + s(percentage_index, by = source)
               + ti(par_valence, percentage_index, by = source)
               + s(bookId, bs = "re"), "mod6")

compareML(mod0, mod1); compareML(mod1, mod2); compareML(mod2, mod3)
compareML(mod3, mod4); compareML(mod4, mod5); compareML(mod5, mod6)

# Explicit test of the interaction: REPORTED, never used to drop ti().
mod6_noti <- fit_ml(avg_comm_valence ~ source
                    + s(par_valence, by = source, k = 20)
                    + s(percentage_index, by = source)
                    + s(bookId, bs = "re"), "mod6_noti")
compareML(mod6_noti, mod6)
print(AIC(mod6_noti, mod6))

# =============================================================================
# Part B: refinement (fREML, discrete) -- ti RETAINED throughout
# =============================================================================

cat("\n\n########## Part B: refinement (fREML) ##########\n")

rhs_base <- paste(
  "source",
  "s(par_valence, by = source, k = 20)",
  "s(percentage_index, by = source)",
  "ti(par_valence, percentage_index, by = source)",
  sep = " + "
)

fit_fr <- function(rhs_extra, name, data = model_df, rho = NULL, weights = NULL) {
  f <- as.formula(paste("avg_comm_valence ~", rhs_base, "+", rhs_extra))
  args <- list(formula = f, data = data, method = "fREML", discrete = TRUE,
               nthreads = n_threads)
  if (!is.null(weights)) args$weights <- weights
  if (!is.null(rho)) { args$rho <- rho; args$AR.start <- data$start_event }
  m <- do.call(bam, args)
  cat("\n---", name, "---\n"); print(summary(m))
  saveRDS(m, op(paste0(name, ".rds")))
  m
}

# ---- B1. Residual autocorrelation -------------------------------------------
mod_base <- fit_fr("s(bookId, bs = 're')", "mod_base")
rho1 <- start_value_rho(mod_base); cat("rho (unweighted):", rho1, "\n")
png(fp("acf_mod_base.png"), width = 800, height = 600); acf_resid(mod_base); dev.off()

# ---- B2. Book-level structure -----------------------------------------------
mod_re    <- fit_fr("s(bookId, bs = 're')", "mod_re", rho = rho1)
mod_fs    <- fit_fr("s(percentage_index, bookId, bs = 'fs', m = 1)", "mod_fs",
                    rho = rho1)
mod_slope <- fit_fr("s(bookId, bs = 're') + s(bookId, par_valence, bs = 're')",
                    "mod_slope", rho = rho1)
mod_fs_slope <- fit_fr(paste("s(percentage_index, bookId, bs = 'fs', m = 1)",
                             "+ s(bookId, par_valence, bs = 're')"),
                       "mod_fs_slope", rho = rho1)

cand <- list(mod_re = mod_re, mod_fs = mod_fs,
             mod_slope = mod_slope, mod_fs_slope = mod_fs_slope)
aic_tab <- AIC(mod_re, mod_fs, mod_slope, mod_fs_slope)
print(aic_tab)
best_name <- rownames(aic_tab)[which.min(aic_tab$AIC)]
cat("Best book-level structure:", best_name, "\n")
mod_best <- cand[[best_name]]
best_formula <- formula(mod_best)

# ---- B3. Weighting ----------------------------------------------------------
# avg_comm_valence averages over comm_count comments, so rows differ in
# precision. Raw counts let a few viral paragraphs dominate the likelihood;
# log1p keeps the precision ordering without the tail domination.
cc <- model_df$comm_count
cat("\ncomm_count: median", median(cc), " max", max(cc), "\n")
for (p in c(0.001, 0.01, 0.05)) {
  thr <- quantile(cc, 1 - p)
  cat(sprintf("top %.1f%% rows (>= %.0f): raw %.1f%% of weight, log1p %.1f%%\n",
              100 * p, thr, 100 * sum(cc[cc >= thr]) / sum(cc),
              100 * sum(log1p(cc[cc >= thr])) / sum(log1p(cc))))
}

# do.call, not a direct bam() call: bam evaluates `weights` inside the data
# frame's environment, so a local variable holding the weight vector is not
# visible to it. do.call substitutes the vector itself into the call.
fit_weighted <- function(data, w, name) {
  base_args <- list(formula = best_formula, data = data, method = "fREML",
                    discrete = TRUE, nthreads = n_threads, weights = w)
  m0 <- do.call(bam, base_args)
  rho <- start_value_rho(m0)
  cat("\n", name, ": rho =", rho, "\n")
  m <- do.call(bam, c(base_args,
                      list(rho = rho, AR.start = data$start_event)))
  cat("\n---", name, "---\n"); print(summary(m))
  saveRDS(m, op(paste0(name, ".rds")))
  m
}

mod_final <- fit_weighted(model_df, log1p(model_df$comm_count), "mod_final")

# =============================================================================
# Part C: sensitivity fits
# =============================================================================

cat("\n\n########## Part C: sensitivity ##########\n")

smooth_terms <- function(m, lab) {
  s <- summary(m)$s.table
  data.frame(model = lab, term = rownames(s), s, row.names = NULL,
             check.names = FALSE)
}
sens <- list(primary = mod_final)

# C1. Raw comm_count weights (expected to be tail-driven)
sens$raw_weights <- fit_weighted(model_df, model_df$comm_count, "mod_final_raww")

# C2. Exact comment anchors only.
#
# The Qidian side reaches paragraphs through the recovered segmentId ->
# paragraph mapping. For the 58 books keeping the U+3000 indent that link is
# exact; for the 49 glued books it is proportional (median error 0 paragraphs,
# 84% within one, 93% within two). Refitting on the exact subset alone is the
# test of whether the approximation matters. If the ti() rows barely move, it
# does not.
if (has_anchor) {
  df_exact <- model_df %>% filter(comm_anchor == 1) %>%
    arrange(bookId, new_index) %>% droplevels()
  df_exact$start_event <- !duplicated(df_exact$bookId)
  cat(sprintf("\nExact-anchor subset: %d of %d rows (%.1f%%), %d books\n",
              nrow(df_exact), nrow(model_df),
              100 * nrow(df_exact) / nrow(model_df),
              nlevels(df_exact$bookId)))
  if (nrow(df_exact) > 1000 && nlevels(df_exact$bookId) > 5) {
    sens$exact_anchor <- fit_weighted(df_exact, log1p(df_exact$comm_count),
                                      "mod_final_exact_anchor")
  } else {
    cat("subset too small to refit; skipped\n")
  }
} else {
  cat("\nNOTE: no `comm_anchor` column. Regenerate the long table with\n",
      "scripts/data-preapration20260906.Rmd to enable the anchor sensitivity.\n",
      sep = "")
}

# C3. Paragraphs with no Chinese text.
#
# Where the segmenter found no Chinese, the Qidian row carries no par_valence
# and is already gone from model_df; what remains is the WebNovel half of
# those paragraphs. Dropping them tests whether the unpaired English stretches
# pull the WebNovel curve. Ordinary unpaired rows -- one platform's readers
# simply not commenting -- are real behaviour and stay in.
if (has_zh_missing) {
  df_zh_ok <- model_df %>% filter(!zh_missing) %>%
    arrange(bookId, new_index) %>% droplevels()
  df_zh_ok$start_event <- !duplicated(df_zh_ok$bookId)
  cat(sprintf("\nzh-present subset: %d of %d rows (%.2f%% dropped)\n",
              nrow(df_zh_ok), nrow(model_df),
              100 * (1 - nrow(df_zh_ok) / nrow(model_df))))
  print(model_df %>% group_by(source) %>%
          summarise(rows = n(), dropped = sum(zh_missing),
                    pct = sprintf("%.2f%%", 100 * mean(zh_missing)),
                    .groups = "drop"))
  sens$zh_present <- fit_weighted(df_zh_ok, log1p(df_zh_ok$comm_count),
                                  "mod_final_zh_ok")
} else {
  cat("\nNOTE: no `zh_missing` column; skipping that sensitivity fit.\n")
}

cat("\n--- smooth terms across all fits (the ti() rows carry the RQ) ---\n")
print(do.call(rbind, Map(smooth_terms, sens, names(sens))))
cat("\n--- parametric terms, primary ---\n"); print(summary(mod_final)$p.table)

# =============================================================================
# Part D: diagnostics on the primary model
# =============================================================================

cat("\n\n########## Part D: diagnostics ##########\n")
png(fp("gam_check_mod_final.png"), width = 800, height = 800)
par(mfrow = c(2, 2)); gam.check(mod_final); dev.off()
png(fp("acf_mod_final.png"), width = 800, height = 600)
acf_resid(mod_final, split_pred = "AR.start"); dev.off()

# par_valence is not detrended, so check it against the progression smooths;
# and the ti() terms may overlap the fs book smooths -- high overlap there
# means the interaction could be absorbing book-level progression
# heterogeneity rather than a real effect.
print(concurvity(mod_final, full = TRUE))
print(concurvity(mod_final, full = FALSE)$estimate)

# =============================================================================
# Part E: linear mixed-model comparison
# =============================================================================

cat("\n\n########## Part E: linear mixed model ##########\n")
# Linear analogue: fixed interactions replace the smooths and ti,
# (1 + par_valence | bookId) replaces the book smooths. Same log1p weights.
# CAVEAT: the GAM also carries an AR(1) working correlation and the two
# likelihoods come from different machinery -- read R2/RMSE, not the AIC.
mod_final_linear <- lmer(
  avg_comm_valence ~ source * par_valence * percentage_index
    + (1 + par_valence | bookId),
  data = model_df, weights = log1p(model_df$comm_count), REML = TRUE
)
print(summary(mod_final_linear))
saveRDS(mod_final_linear, op("mod_final_linear.rds"))
print(compare_performance(mod_final_linear, mod_final))

# =============================================================================
# Part F: figures
# =============================================================================

cat("\n\n########## Part F: figures ##########\n")

# Book-level terms to exclude for population-level prediction, read off the
# fitted model rather than hard-coded: their labels depend on which structure
# won in B2, and a stale literal would silently predict the wrong thing.
book_terms <- Filter(function(l) grepl("bookId", l),
                     vapply(mod_final$smooth, function(s) s$label, character(1)))
cat("excluding book-level terms:", paste(book_terms, collapse = ", "), "\n")

quants <- quantile(model_df$par_valence, c(0.1, 0.5, 0.9), na.rm = TRUE)
names(quants) <- c("Negative", "Neutral", "Positive")
print(quants)

pi_seq <- seq(min(model_df$percentage_index), max(model_df$percentage_index),
              length.out = 200)
pred_grid <- expand.grid(percentage_index = pi_seq,
                         source = levels(model_df$source),
                         par_valence = quants,
                         bookId = model_df$bookId[1])
pred_grid$sentiment_level <- factor(
  names(quants)[match(pred_grid$par_valence, quants)],
  levels = c("Negative", "Neutral", "Positive"))

preds <- predict(mod_final, newdata = pred_grid, se.fit = TRUE,
                 exclude = book_terms)
pred_grid$fit <- preds$fit
pred_grid$lower <- preds$fit - 1.96 * preds$se.fit
pred_grid$upper <- preds$fit + 1.96 * preds$se.fit

pal <- c("Negative" = "#4A90D9", "Neutral" = "#E8A838", "Positive" = "#D94A4A")

p_gam <- ggplot(pred_grid, aes(percentage_index, fit,
                               color = sentiment_level, fill = sentiment_level)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1) + facet_wrap(~ source) +
  scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "GAM Predictions with 95% Confidence Intervals",
       subtitle = "Population level (book effects excluded); log1p(comm_count)-weighted fit",
       x = "Story progression", y = "Predicted comment valence",
       color = "Paragraph valence", fill = "Paragraph valence") +
  theme_minimal() +
  theme(legend.position = "bottom",
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA))
ggsave(fp("GAM_predictions.png"), p_gam, width = 10, height = 5, dpi = 300)

# Observed vs predicted
terciles <- quantile(model_df$par_valence, c(1/3, 2/3), na.rm = TRUE)
observed_means <- model_df %>%
  mutate(sentiment_level = factor(case_when(
           par_valence < terciles[1] ~ "Negative",
           par_valence <= terciles[2] ~ "Neutral",
           TRUE ~ "Positive"),
           levels = c("Negative", "Neutral", "Positive")),
         pct_bin = round(percentage_index * 2) / 2) %>%
  group_by(source, sentiment_level, pct_bin) %>%
  summarise(mean_comm = mean(avg_comm_valence, na.rm = TRUE),
            se = sd(avg_comm_valence, na.rm = TRUE) / sqrt(n()),
            lower = mean_comm - 1.96 * se, upper = mean_comm + 1.96 * se,
            .groups = "drop")

common_theme <- theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 9),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA))

p_obs <- ggplot(observed_means, aes(pct_bin, mean_comm,
                                    color = sentiment_level, fill = sentiment_level)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1, color = NA) +
  geom_line(linewidth = 0.5) + facet_wrap(~ source) +
  scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "A. Observed Comment Valence",
       subtitle = "Binned means with 95% CIs; valence terciles",
       x = "Story progression", y = "Comment valence") + common_theme

p_pred <- ggplot(pred_grid, aes(percentage_index, fit,
                                color = sentiment_level, fill = sentiment_level)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.1, color = NA) +
  geom_line(linewidth = 0.5) + facet_wrap(~ source) +
  scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "B. GAM-Predicted Comment Valence",
       subtitle = "Predictions at the 10th/50th/90th percentile of paragraph valence",
       x = "Story progression", y = "Comment valence") + common_theme

combined <- (p_obs / p_pred) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
combined <- combined + plot_annotation(
  caption = "Note: shaded areas are 95% CIs. GAM predictions exclude book-level effects.",
  theme = theme(plot.caption = element_text(size = 9, hjust = 0, face = "italic")))
ggsave(fp("observed_vs_predicted.png"), combined, width = 12, height = 10, dpi = 300)

# Difference plot: 90th - 10th percentile of par_valence.
#
# The CIs above are for the LEVEL of each line and carry the large shared
# intercept + progression + book uncertainty. That cancels in a difference, so
# the contrast is estimated far more precisely than the overlapping bands
# suggest. Because ti() depends on par_valence its columns do NOT cancel: the
# line is flat only if the interaction is genuinely absent. This is the figure
# that answers the research question.
diff_grid <- expand.grid(percentage_index = pi_seq,
                         source = levels(model_df$source))
# unname(): a named value here makes transform() try to take row names from
# it and warn on every call
g_hi <- transform(diff_grid, par_valence = unname(quants["Positive"]),
                  bookId = model_df$bookId[1])
g_lo <- transform(diff_grid, par_valence = unname(quants["Negative"]),
                  bookId = model_df$bookId[1])
Xd <- predict(mod_final, newdata = g_hi, type = "lpmatrix") -
      predict(mod_final, newdata = g_lo, type = "lpmatrix")
b <- coef(mod_final); Vb <- vcov(mod_final)
diff_grid$diff  <- as.vector(Xd %*% b)
diff_grid$se    <- sqrt(rowSums((Xd %*% Vb) * Xd))
diff_grid$lower <- diff_grid$diff - 1.96 * diff_grid$se
diff_grid$upper <- diff_grid$diff + 1.96 * diff_grid$se

p_diff <- ggplot(diff_grid, aes(percentage_index, diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "#4A4A4A") +
  geom_line(linewidth = 0.8) + facet_wrap(~ source) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "Effect of paragraph valence on comment valence over story progression",
       subtitle = "Predicted difference (90th - 10th percentile of par_valence), 95% CI, from the ti() model",
       x = "Story progression", y = "Difference in comment valence") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA))
ggsave(fp("valence_effect_difference.png"), p_diff, width = 10, height = 4.5, dpi = 300)

write_csv(diff_grid, op("valence_effect_difference.csv"))

cat("\nDone in", round(difftime(Sys.time(), t_start, units = "mins"), 1),
    "minutes\n")
sessionInfo()
