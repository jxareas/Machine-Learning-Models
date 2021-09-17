library(dplyr)
library(ggplot2)
library(car)
library(emmeans)
library(effects)
library(ggthemes)

df <-
  as.data.frame(read.csv("./data/BaumannJones.csv", stringsAsFactors = T))

# Analyzing the dataframe -----------------------------------------------------------------------------------------

summary(df)

# The experimental design has 22 subjects in each group
xtabs(~group, df)

# Visualizing the difference in means & standard deviation
df %>%
  select(group, post.test.3) %>%
  group_by(group) %>%
  summarize("mean" = mean(post.test.3), "sd" = sd(post.test.3))

# Plotting the Variability and Difference in Means between Groups ----------------
ggplot(data = df, mapping = aes(x = group, y = post.test.3,
                                fill = group)) +
  stat_boxplot(geom = "errorbar", width = 0.8, alpha = 0.5) +
  geom_boxplot() +
  stat_summary(fun = mean, size = 1, color = "red",
               mapping = aes(shape = "Mean")) +
  labs(
    title = "Variability between Groups",
    x = "Group",
    y = "Third Post Test Score",
    shape = "Statistic"
  ) +
  scale_fill_brewer(palette = "Set3") +
  theme_economist_white() +
  theme(
    plot.title = element_text(face = "bold", hjust = .5, vjust = 2),
    axis.title.x = element_text(face = "bold", color = "dimgrey", vjust = -1),
    axis.title.y = element_text(face = "bold", color = "dimgrey", vjust = 3),
    legend.text = element_text(size = 10),
    legend.position = "right"
  ) +
  guides(fill = "none")

# Fitting a One-Way ANOVA Linear Model -------------------------------------

fit_anova <-
  lm(
    formula = post.test.3 ~ group,
    data = df
  )

fit_summary <-
  S(fit_anova)

# The coefficients are estimates for the difference in means
# with respect to the reference category, the mean score of the
# Basal Group
coef(fit_anova)

# Pairwise Difference in Means Comparisons ----------------------------------

# Estimated Means, their Confidence Intervals &
# Difference in Means Hypothesis Tests
emmeans(fit_anova, pairwise ~ group)
# Only the Difference in (Score) Means between the Basal & DRTA
# groups is significantly different from 0.

# Plotting the Difference in Means & Variability as a Predictor Effect -------------------------------------

# Plotting the Estimated Mean & their Standard Errors
plot(predictorEffects(fit_anova),
     main = "Estimated 3rd Post-Test Mean Score", sub = "Group Predictor Effect Plot",
     ylab = "3rd Post-Test Mean Score", xlab = "Group")