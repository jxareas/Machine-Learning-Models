library(dplyr)
library(car)
library(ggplot2)
library(ggfortify)
library(ggthemes)

df <-
  as.data.frame(read.csv("./data/Davis.csv"))
# Plotting the data ----------------------------------------------------------

ggplot(data = df, mapping = aes(x = repwt, y = weight, col = weight)) +
  geom_jitter(size = 2) +
  geom_smooth(method = lm, formula = y ~ x,
              color = "green", size = 1) +
  labs(
    title = "Simple Linear Regression Fit",
    subtitle = "Weight (kg) vs Reported Weight (kg)",
    x = "Reported Weight",
    y = "Weight"
  ) +
  scale_color_gradient(low="blue", high = "red") +
  theme(
    plot.title = element_text(face = "bold", hjust = .5),
    plot.subtitle = element_text(face = "italic", hjust = .5),
  ) +
  guides(col = "none")

# Fitting the Linear Model ---------------------------------------------------

# Reported Weight ~ Weight, so we expect approximately b0 = 0 and b1 = 1
fit <-
  lm(
    formula = weight ~ repwt,
    data = df
  )

sfit <-
  S(fit)

# Confidence & Prediction Intervals ---------------------------------------

# Coefficients and their 95% Confidence Interval
print(Confint(fit, estimate = T, level = 0.95))

# Coefficients and their 95% Prediction Interval
print(predict(object=fit,
        newdata = data.frame("repwt" = 93),
        interval = "prediction",
        alpha = 0.05))

# Influence Measures --------------------------------------------------------
influences <- data.frame(
  hatvalues(fit), dffits(fit)
)

names(influences) <- c("hats", "dffits")

# Top observations by influence
influences %>% arrange(desc(dffits)) %>% select(dffits) %>% head()

# Top observations by leverage
influences %>% arrange(desc(hats)) %>% select(hats) %>% head()

# Removing the value with high influence and updating the model ------------------------------

fit2 <-
  lm(
    formula = weight ~ repwt,
    data = df,
    subset = -12
  )

sfit2 <-
  S(fit2)

# Comparing the Coefficients between the Linear Models --------------------------

print(compareCoefs(fit, fit2))

# Linear Hypothesis ----------------------------------------------------

print(linearHypothesis(fit2, "repwt = 1"))

# Diagnostic Plots for the Second Fit (Removed Outlier) ---------------------------------------------------------------

autoplot(fit2, which = 1:6, colour = "darkblue",
         smooth.colour = "magenta", ad.colour = "black",
         label.n = 0, ncol = 3, size = 2
         ) +
  theme_stata() +
  theme(
    plot.title = element_text(face = "bold", hjust = .5),
    axis.title.x = element_text(face = "bold", color = "dimgrey"),
    axis.title.y = element_text(face = "bold", color = "dimgrey"),
  )
