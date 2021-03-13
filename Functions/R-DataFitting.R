library(tidyverse)
library(zoo)
library(CVXR)

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

write.csv(data, "E:/Network/MFM_Classes/5031/Copulas/Project/Code/Probability of Default/use_data.csv")

c_bs <- unique(data$CREDIT_BUCKET)

for( c in 1:length(c_bs)){
  
  data[paste("A_", c_bs[c], "_0", sep = "")] <- 0
  data[paste("A_", c_bs[c], "_1", sep = "")] <- 0
  
  data[(data$CREDIT_BUCKET == c_bs[c] & data$NEW_LOAN == 0), paste("A_", c_bs[c], "_0", sep = "")] <- 1
  data[(data$CREDIT_BUCKET == c_bs[c] & data$NEW_LOAN == 1), paste("A_", c_bs[c], "_1", sep = "")] <- 1
  
}

dts <- unique(data$QTR)

for(d in 1:length(dts)){
  data[paste("B_", dts[d], sep = "")] <- 0
  
  data[data$QTR == dts[d],paste("B_", dts[d], sep = "")] <- 1
}

indep <- data[,!(names(data) %in% c("y","DATE", "PD", "PD1", "QTR","CREDIT_BUCKET","NEW_LOAN","CURR_LOANS","BAD_LOANS","total_volume","PD"))]
alphs <- indep[,!grepl("B_", names(indep))]

data["PD1"] <- data["PD"]
data[data["PD"] == 0, "PD1"] <- 0.000000000001
data["y"] <- lapply(data["PD1"], qnorm)


mX <- data.matrix(indep)

vY <- data.matrix(data["y"])

b <- Variable(length(indep))

obj <- Minimize(sum((vY - mX %*% b)^2))

constraints <- list(
                    b[23] + b[24] + b[25] + b[26] + b[27] + b[28] +
                    b[29] + b[30] + b[31] + b[32] + b[33] + b[34] +
                    b[35] + b[36] + b[37] + b[38] + b[39] + b[40] + 
                    b[41] + b[42] + b[43] + b[44] + b[45] + b[46] +
                    b[47] + b[48] + b[49] + b[50] + b[51] + b[52] +
                    b[53] + b[54] + b[55] == 0
                  )

problem <- Problem(obj, constraints)

soln <- solve(problem)

vBetaTS = soln$getValue(b)

names(vBetaTS) <- names(indep)

out_data <- merge(data.frame(CREDIT_BUCKET = c_bs), data.frame(NEW_LOAN = c(0,1)), all = TRUE)

out_data$norm_rho <- mean(vBetaTS[grepl("B_",names(vBetaTS))] * vBetaTS[grepl("B_",names(vBetaTS))] ) / (1 + mean(vBetaTS[grepl("B_",names(vBetaTS))] * vBetaTS[grepl("B_",names(vBetaTS))]))
out_data$norm_PD <- 0

for(row in 1:nrow(out_data)){
    out_data[row,"norm_PD"] <- pnorm(vBetaTS[paste("A_",out_data[row,"CREDIT_BUCKET"],"_", out_data[row,"NEW_LOAN"], sep = "")] * sqrt(1 - out_data[row,"norm_rho"]))
}

risk_factor <- -vBetaTS[grepl("B_",names(vBetaTS))] / sqrt(mean(vBetaTS[grepl("B_",names(vBetaTS))] ^ 2))

acf(risk_factor)

names(risk_factor) <- substr(names(risk_factor), 3, nchar(names(risk_factor)))

risk_plot <- data.frame(list(date = as.yearqtr(names(risk_factor), format = "%Y-Q%q"), risk_factor = risk_factor))


ggplot(risk_plot, aes(x = date, y = risk_factor)) + 
  ggtitle("Risk Factor Over Time") +
  xlab("Date") +
  ylab("Value") +
  geom_line()

ggsave("E:/Network/MFM_Classes/5031/Copulas/Project/Code/Model Results/risk_factor_timeseries.png", dpi = 600)

data$Estimated <- pnorm((mX %*% vBetaTS))


for( c in 1:(length(c_bs) - 1)){
  
  tmp <- data %>% filter(CREDIT_BUCKET == c_bs[c]) %>%
               select(DATE, NEW_LOAN, PD, Estimated) %>%
               rename(Actual = PD)
  
  tmp$Loan_Age <- ifelse(tmp$NEW_LOAN == 1, "New Loan", "Old Loan")
  
  ggplot(tmp) +
    facet_grid(rows = vars(Loan_Age)) +
    geom_line(aes(x = DATE, y = Estimated, color = "Estimated")) +
    geom_point(aes(x = DATE, y = Actual, color = "Actual")) +
    ggtitle(paste("Credit Bucket Model Comparison - ", c_bs[c], sep = "")) +
    xlab("Date") +
    ylab("Probability of Default") +
    scale_y_continuous(labels = scales::percent)
  
  ggsave(paste("E:/Network/MFM_Classes/5031/Copulas/Project/Code/Model Results/Comparison/", str_replace(str_replace(c_bs[c], "<", "lt_"), ">","gt_"), "_normal.png", sep = "" ), dpi = 600)
  
}

