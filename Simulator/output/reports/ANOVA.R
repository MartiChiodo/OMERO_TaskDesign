# Packages
library(tidyr)        # wide -> long
library(dplyr)        # data manipulation
library(effectsize)   # eta squared


# 1. Read the two CSVs (wide format: scenarios in rows, seeds in columns)

path_policy <- "Simulator/output/reports/csv/Opt_False_throughput.csv"
path_opt    <- "Simulator/output/reports/csv/Opt_True_throughput.csv"

# check.names = FALSE keeps the seed column names as they are
policy_wide <- read.csv(path_policy, check.names = FALSE)
opt_wide    <- read.csv(path_opt,    check.names = FALSE)


# 2. Reshape from wide to long

policy_long <- pivot_longer(policy_wide, cols = -Scenario,
                            names_to = "seed", values_to = "thr_policy")
opt_long    <- pivot_longer(opt_wide,    cols = -Scenario,
                            names_to = "seed", values_to = "thr_opt")


# 3. Pair the two strategies and compute the difference

# match rows sharing the same Scenario and seed
diffs <- merge(policy_long, opt_long, by = c("Scenario", "seed"))

# difference: optimizer minus policy (D > 0 means optimizer is better)
diffs$D <- diffs$thr_opt - diffs$thr_policy

nrow(diffs)   # expect one row per scenario-seed pair


# 4. Decode the three factors from the scenario ID

diffs$d1 <- diffs$Scenario %/% 10   # first digit
diffs$d2 <- diffs$Scenario %% 10    # second digit

# warehouse size from the first digit
diffs$warehouse_size <- NA
diffs$warehouse_size[diffs$d1 == 1] <- "small"
diffs$warehouse_size[diffs$d1 == 3] <- "medium"
diffs$warehouse_size[diffs$d1 == 5] <- "large"

# order lines from the second digit (odd = few, even = many)
diffs$order_lines <- NA
diffs$order_lines[diffs$d2 %% 2 == 1] <- "few"
diffs$order_lines[diffs$d2 %% 2 == 0] <- "many"

# arrival rate from the second digit (low for the first two, high for the last two)
diffs$arrival_rate <- NA
diffs$arrival_rate[diffs$d2 <= 2] <- "low"
diffs$arrival_rate[diffs$d2 >= 3] <- "high"


# 5. Declare the factors (fix the level order)

diffs$warehouse_size   <- factor(diffs$warehouse_size,   levels = c("small", "medium", "large"))
diffs$order_lines <- factor(diffs$order_lines, levels = c("few", "many"))
diffs$arrival_rate     <- factor(diffs$arrival_rate,     levels = c("low", "high"))

# sanity check: verify the decoding against the configuration table
# expect each scenario mapped to the correct (warehouse_size, order_lines, arrival_rate)
unique(diffs[, c("Scenario", "warehouse_size", "order_lines", "arrival_rate")])


# 6. Descriptive statistics: mean advantage per scenario

# expect D positive across scenarios if the optimizer is consistently better
aggregate(D ~ warehouse_size + order_lines + arrival_rate, data = diffs,
          FUN = function(x) c(mean = mean(x), sd = sd(x)))


# 7. Main hypothesis: does the optimizer beat the policy overall?

# paired difference already computed, so a one-sample t-test on D
# expect a small p-value and a positive mean if the optimizer wins
t.test(diffs$D)


# 8. ANOVA on the differences: which factors affect the advantage?

model <- lm(D ~ warehouse_size * order_lines * arrival_rate, data = diffs)
summary(model)

# expect significant terms for the factors that change the advantage
anova(model)

# effect size: expect larger eta^2 for the more influential factors
eta_squared(model, partial = FALSE)


# 9. Assumption checks (on the model residuals)

# normality: expect points close to the line, and a non-significant Shapiro test
pdf("qqplot.pdf", width = 5, height = 5)
qqnorm(residuals(model)); qqline(residuals(model), col = "red")
dev.off()
shapiro.test(residuals(model))

# constant variance: expect a shapeless cloud around zero, no funnel
pdf("residuals.pdf", width = 5, height = 5)
plot(fitted(model), residuals(model),
     xlab = "Fitted values", ylab = "Residuals"); abline(h = 0, lty = 2)
dev.off()
