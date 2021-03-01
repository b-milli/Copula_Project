library(tidyverse)
library(zoo)

in_data <- read.csv("E:/Network/MFM_Classes/5031/Copulas/Project/Code/Probability of Default/ALL LOANS Buckets.csv", stringsAsFactors=FALSE)

in_data$DATE <- as.Date(as.yearqtr(in_data$QTR, format = "%Y-Q%q"))

data <- in_data %>% filter(CREDIT_BUCKET != 'No Score', !grepl('2010', QTR, fixed = TRUE), !grepl('2011', QTR, fixed = TRUE))


data$CREDIT_BUCKET <- ifelse(data$CREDIT_BUCKET %in% c("820-839", ">840"), ">820", ifelse(data$CREDIT_BUCKET %in% c("<600", "600-619", "620-639"), "<640", data$CREDIT_BUCKET))

data <- data %>% group_by(QTR, DATE, CREDIT_BUCKET, NEW_LOAN) %>%
                 summarise(
                   CURR_LOANS = sum(CURR_LOANS),
                   BAD_LOANS = sum(BAD_LOANS),
                   total_volume = sum(total_volume)
                 ) %>%
                ungroup()

data$PD <- data$BAD_LOANS / data$CURR_LOANS

c_bs <- unique(data$CREDIT_BUCKET)

for( c in 1:(length(c_bs) - 1)){
  
  data[paste("A_", c_bs[c], "_0", sep = "")] <- 0
  data[paste("A_", c_bs[c], "_1", sep = "")] <- 0
  
  data[(data$CREDIT_BUCKET == c_bs[c] & data$NEW_LOAN == 0), paste("A_", c_bs[c], "_0", sep = "")] <- 1
  data[(data$CREDIT_BUCKET == c_bs[c] & data$NEW_LOAN == 1), paste("A_", c_bs[c], "_1", sep = "")] <- 1
  
}

dts <- unique(data$QTR)

for(d in 1:(length(dts) - 1)){
  data[paste("B_", dts[d], sep = "")] <- 0
  
  data[data$QTR == dts[d],paste("B_", dts[d], sep = "")] <- 1
}

indep <- data[,!(names(data) %in% c("y", "PD", "PD1", "QTR","CREDIT_BUCKET","NEW_LOAN","CURR_LOANS","BAD_LOANS","total_volume","PD"))]
alphs <- indep[,!grepl("B_", names(indep))]

data["PD1"] <- data["PD"]
data[data["PD"] == 0, "PD1"] <- 0.000000000001
data["y"] <- lapply(data["PD1"], qnorm)

mX <- cbind("int"=1,data.matrix(indep))
mZ <- cbind("int"=1,data.matrix(alphs))
vY <- data.matrix(data["y"])

vBetaOLS = coef(lmHetMean <- lm.fit(y = vY, x = mX))



residHet = resid(lmHetMean)
vVarEst = exp(fitted(lmHetVar <- lm.fit(y = log(residHet^2), x = mZ)))

vBetaTS = coef(GLS <- lm.fit(y = vY/vVarEst, x = apply(mX, 2, function(x) x/vVarEst)))

data$Estimated <- pnorm(fitted(GLS) * vVarEst)

for( c in 1:(length(c_bs) - 1)){
  
  tmp <- data %>% filter(CREDIT_BUCKET == c_bs[c]) %>%
               select(DATE, NEW_LOAN, PD, Estimated) %>%
               rename(Actual = PD)
  
  tmp$Loan_Age <- ifelse(tmp$NEW_LOAN == 1, "New Loan", "Old Loan")
  
  ggplot(tmp) +
    facet_grid(rows = vars(Loan_Age)) +
    geom_line(aes(x = DATE, y = Estimated, color = "red")) +
    geom_point(aes(x = DATE, y = Actual, color = "blue")) +
    ggtitle(paste("Credit Bucket Model Comparison - ", c_bs[c], sep = ""))
  
  ggsave(paste("E:/Network/MFM_Classes/5031/Copulas/Project/Code/Model Results/Plots/R Comparison/", str_replace(str_replace(c_bs[c], "<", "lt_"), ">","gt_"),".png", sep = "" ), dpi = 600)
  
}
