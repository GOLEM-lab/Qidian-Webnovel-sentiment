library(mgcv) # for the GAM model
library(readr)
library(itsadug)

long_df_roll <- read_csv("/data/long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv")

long_df_roll$bookId <- as.factor(long_df_roll$bookId)
long_df_roll$source <- as.factor(long_df_roll$source)

# 1. Base: just paragraph sentiment
mod1 <- gam(comm_sentiment_roll_51 ~ s(par_sentiment_roll_51),
            data = long_df_roll, method = "ML")

print(summary(mod1))

# 2. Add story progression (additive)
mod2 <- gam(comm_sentiment_roll_51 ~ s(par_sentiment_roll_51) 
            + s(percentage_index),
            data = long_df_roll, method = "ML")
            
print(summary(mod2))
saveRDS(mod2, "mod2.rds")

# 3. Add interaction between sentiment and progression
mod3 <- gam(comm_sentiment_roll_51 ~ s(par_sentiment_roll_51) 
            + s(percentage_index) 
            + ti(par_sentiment_roll_51, percentage_index),
            data = long_df_roll, method = "ML")

print(summary(mod3))
saveRDS(mod3, "mod3.rds")

# 4. Add source as intercept shift only
mod4 <- gam(comm_sentiment_roll_51 ~ source 
            + s(par_sentiment_roll_51) 
            + s(percentage_index) 
            + ti(par_sentiment_roll_51, percentage_index),
            data = long_df_roll, method = "ML")

print(summary(mod4))
saveRDS(mod4, "mod4.rds")

# 5. Allow all smooths to vary by source
mod5 <- gam(comm_sentiment_roll_51 ~ source 
            + s(par_sentiment_roll_51, by = source) 
            + s(percentage_index, by = source) 
            + ti(par_sentiment_roll_51, percentage_index, by = source),
            data = long_df_roll, method = "ML")

print(summary(mod5))
saveRDS(mod5, "mod5.rds")

# 6. Add random intercept per book
mod6 <- gam(comm_sentiment_roll_51 ~ source 
            + s(par_sentiment_roll_51, by = source) 
            + s(percentage_index, by = source) 
            + ti(par_sentiment_roll_51, percentage_index, by = source) 
            + s(bookId, bs = "re"),
            data = long_df_roll, method = "ML")

print(summary(mod6))
saveRDS(mod6, "mod6.rds")

gam.check(mod6)

compareML(mod1, mod2)
compareML(mod2, mod3)
compareML(mod3, mod4)
compareML(mod4, mod5)
compareML(mod5, mod6)

