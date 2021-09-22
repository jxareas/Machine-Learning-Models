library(ggplot2)
library(effects)
library(car)
library(dplyr)
library(reshape2)
library(ggthemes)

df <-
  as.data.frame(read.csv(file="./data/Prestige.csv"))

# Fitting a Linear Model -----------------------------------------------------------------------------------------

fit <-
  lm(formula = prestige ~ education + log2(income) + women, data = df)

# We can look in the summary that the Women Variable is not statistically significant
fit_summary <-
  S(fit)

# Regression Coefficients Plot ------------------------------------------------------------------------------------

# Selecting all the predictors & excluding the intercept
conf_int <- Confint(fit)[-1, ] %>% as.data.frame()

conf_int <- conf_int %>%
  mutate(predictor = row.names(conf_int), .before = Estimate)

row.names(conf_int) <- NULL

names(conf_int) <- c("predictor", "estimate", "lower", "upper")

ggplot(data = conf_int,
       mapping = aes(x = predictor, y = estimate)
) + geom_point(color = "red", size = 2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .05) +
  labs(
    title = "Regression Coefficients & 95% Confidence Intervals",
    x = "Predictor",
    y = "Coefficient"
  ) +
  theme_stata() +
  theme(
    plot.title = element_text(face = "bold", hjust = .5),
    axis.title.x = element_text(face = "bold", vjust = -1,
                                color = "dimgrey"),
    axis.title.y = element_text(face = "bold", vjust = 3,
                                color = "dimgrey"),
  )

# Predictor Effects ---------------------------------------------

plot(predictorEffects(fit))

# Correlation Matrix ---------------------------------------------
correlation_matrix <- cor(Filter(is.numeric, df))
correlation_matrix[upper.tri(correlation_matrix)] <- NA
correlation_matrix <- melt(correlation_matrix, varnames = c("x", "y"), value.name = "pearson", na.rm = T)

ggplot(data = correlation_matrix, mapping = aes(x, y, fill = pearson)) +
  geom_tile() +
  labs(
    title = "Correlation Matrix",
    x = "",
    y = "",
    fill = "Pearson Correlation"
  ) +
  scale_fill_gradient2(low = "red", mid = "white", high = "darkblue",
                 limit = c(-1, 1), breaks = c(-1, -.5, 0, .5, 1)) +
  theme(
    plot.title = element_text(face = "bold", hjust = .5),
    legend.title = element_text(face = "italic",
                                vjust = 3)
  )

