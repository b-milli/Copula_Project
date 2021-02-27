
library(tidyverse)

in_data <- read.csv("../project_data.csv", colClasses = 
                      c("numeric", "character", "character", "character", "character", "character", "character",
                        "character", "numeric", "numeric", "numeric", "numeric",
                        "numeric", "numeric", "character", "character", "numeric", "numeric",
                        "numeric", "character", "numeric", "numeric", "character", "numeric",
                        "numeric", "numeric", "character", "character", "character",
                        "numeric", "character", "character", "character", "character",
                        "numeric", "character", "character", "character", "character",
                        "numeric", "character", "character", "character", "character",
                        "character", "character", "numeric", "character", "numeric",
                        "numeric", "numeric", "character", "character", "character",
                        "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                        "numeric", "numeric", "numeric", "numeric", "numeric", "character",
                        "numeric", "character", "numeric", "numeric", "numeric", "numeric",
                        "numeric", "numeric", "character", "numeric", "numeric", "numeric",
                        "numeric", "character", "numeric", "character", "numeric", "character",
                        "numeric", "numeric", "character", "character", "numeric", "numeric",
                        "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                        "numeric", "numeric", "numeric", "numeric", "numeric", "character",
                        "character", "character", "character", "character",
                        "character", "numeric", "numeric"))

in_data$PRCS_DTE <- as.Date(paste(substr(in_data$ACT_PERIOD, 3,nchar(in_data$ACT_PERIOD)),substr(in_data$ACT_PERIOD,1,2), "01", sep = "-"), format = "%Y-%m-%d")
in_data$YEAR <- as.character(lubridate::year(in_data$PRCS_DTE))
in_data$QTR <- as.character(lubridate::quarter(in_data$PRCS_DTE))

in_data$new_loan <- ifelse(in_data$LOAN_AGE <= 24, 1, 0)

tmp <- in_data %>% select(LOAN_ID, PRCS_DTE,  new_loan, CSCORE_B, DLQ_STATUS, CURR_UPB, FORECLOSURE_DATE, ZERO_BAL_CODE) %>%
  mutate(
    cur_qtr = paste(lubridate::year(PRCS_DTE), "-Q", lubridate::quarter(PRCS_DTE), sep = ""),
    lst_qtr = ifelse(lubridate::quarter(PRCS_DTE) == 1, paste(lubridate::year(PRCS_DTE) - 1, "-Q4", sep = ""), paste(lubridate::year(PRCS_DTE), "-Q", lubridate::quarter(PRCS_DTE) - 1, sep = ""))
  ) %>%
  group_by(LOAN_ID, cur_qtr) %>%
  arrange(PRCS_DTE) %>%
  mutate(rn = row_number()) %>%
  filter(rn == 1) %>%
  select(-rn) %>%
  ungroup()

tmp2 <- tmp %>% select(lst_qtr, LOAN_ID, DLQ_STATUS, CURR_UPB, FORECLOSURE_DATE, ZERO_BAL_CODE) %>%
  rename(
    cur_qtr = lst_qtr,
    NEW_BAL = CURR_UPB,
    NEW_DLQ = DLQ_STATUS,
    NEW_FORECLOSURE = FORECLOSURE_DATE,
    NEW_ZERO_BAL = ZERO_BAL_CODE
  )


perform <- merge(tmp, tmp2, by = c("LOAN_ID","cur_qtr"), all.x = TRUE)

perform <- perform %>% filter(CURR_UPB > 0, cur_qtr != "2020-Q2")

peform <- perform %>% filter(as.numeric(DLQ_STATUS) < 3)

perform$CREDIT_BUCKET <- cut(perform$CSCORE_B, seq(600, 860, by = 20), include.lowest = TRUE)
perform$CREDIT_BUCKET <- ifelse(is.na(perform$CREDIT_BUCKET), "No Score", as.character(perform$CREDIT_BUCKET))

perform$NEW_DLQ <- ifelse(perform$NEW_BAL == 0, ifelse(perform$NEW_ZERO_BAL %in% c("97", "98"), 999, 0), perform$NEW_DLQ)
perform$NEW_DLQ <- ifelse(is.na(perform$NEW_DLQ), 999, perform$NEW_DLQ)
perform$DELINQ_IND <- ifelse(as.numeric(perform$NEW_DLQ) >= 3 , 1, 0)
perform$DELINQ_IND <- ifelse(perform$NEW_FORECLOSURE != "" &  perform$FORECLOSURE_DATE != "", 1, perform$DELINQ_IND)

perform <- perform %>% select(-lst_qtr)

write.csv(perform, "E:/Network/MFM_Classes/5031/Copulas/Project/Data/final_qtrly_joined.csv")
