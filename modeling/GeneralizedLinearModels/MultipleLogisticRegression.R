library(dplyr)
library(ggplot2)
library(car)
library(effects)

df <-
  as.data.frame(read.csv("./data/CowlesDavis.csv", stringsAsFactors = T)) %>%
    rename(volunteering = volunteer)

# Fitting the Binary Logistic Model -----------------------------------------

fit <-
  glm(
    formula = volunteering ~ sex + neuroticism * extraversion,
    data = df,
    family = binomial(link = "logit")
  )

fit_summary <-
  S(fit)

# Predictor Effects Plot ---------------------------------------------------

# Neuroticism & Extraversion are in the scale of Eysenck Personality Inventory,
# they are integers potentially ranging from 0 t0 24.
eysenck_scale <- seq(0, 24, by = 8)

plot(predictorEffects (fit, ~ neuroticism + extraversion,
xlevels=list(neuroticism=eysenck_scale,
extraversion=eysenck_scale)), lines=list (multiline=TRUE))
# Plot 1:
# Volunteering is positively related to neuroticism when extraversion is low
# and negatively related when extraversion is high.

# Plot 2:
# Volunteering is generally positively related to extraversion to neuroticism,
# except when neuroticism is at it's highest (24).

# Analysis of Deviance for a Binary Regression Model -------------------------------------

fit_no_interaction <- update(fit,
                             formula = . ~ . -neuroticism:extraversion,
)

# Comparing the Coefficients of both regression models
compareCoefs(fit, fit_no_interaction)

# Nested Model Testing using the Likelihood-Ratio Chi-Square Test
anova(fit_no_interaction, fit, test = "Chisq")

# Low p-value, we prefer the logistic regression model with an interaction
# term.