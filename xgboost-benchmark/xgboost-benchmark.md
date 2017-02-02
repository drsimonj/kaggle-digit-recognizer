Kaggle Digit Recognizer: XGBoost Benchmark
================

This notebook demonstrates a simple benchmark model using extreme gradient boosting from the xgboost package.

``` r
library(xgboost)
library(readr)
library(dplyr)

# Import data
train <- read_csv("../data/train.csv", progress = FALSE)
test  <- read_csv("../data/test.csv", progress = FALSE)

head(train)
```

    ## # A tibble: 6 × 785
    ##   label pixel0 pixel1 pixel2 pixel3 pixel4 pixel5 pixel6 pixel7 pixel8
    ##   <int>  <int>  <int>  <int>  <int>  <int>  <int>  <int>  <int>  <int>
    ## 1     1      0      0      0      0      0      0      0      0      0
    ## 2     0      0      0      0      0      0      0      0      0      0
    ## 3     1      0      0      0      0      0      0      0      0      0
    ## 4     4      0      0      0      0      0      0      0      0      0
    ## 5     0      0      0      0      0      0      0      0      0      0
    ## 6     0      0      0      0      0      0      0      0      0      0
    ## # ... with 775 more variables: pixel9 <int>, pixel10 <int>, pixel11 <int>,
    ## #   pixel12 <int>, pixel13 <int>, pixel14 <int>, pixel15 <int>,
    ## #   pixel16 <int>, pixel17 <int>, pixel18 <int>, pixel19 <int>,
    ## #   pixel20 <int>, pixel21 <int>, pixel22 <int>, pixel23 <int>,
    ## #   pixel24 <int>, pixel25 <int>, pixel26 <int>, pixel27 <int>,
    ## #   pixel28 <int>, pixel29 <int>, pixel30 <int>, pixel31 <int>,
    ## #   pixel32 <int>, pixel33 <int>, pixel34 <int>, pixel35 <int>,
    ## #   pixel36 <int>, pixel37 <int>, pixel38 <int>, pixel39 <int>,
    ## #   pixel40 <int>, pixel41 <int>, pixel42 <int>, pixel43 <int>,
    ## #   pixel44 <int>, pixel45 <int>, pixel46 <int>, pixel47 <int>,
    ## #   pixel48 <int>, pixel49 <int>, pixel50 <int>, pixel51 <int>,
    ## #   pixel52 <int>, pixel53 <int>, pixel54 <int>, pixel55 <int>,
    ## #   pixel56 <int>, pixel57 <int>, pixel58 <int>, pixel59 <int>,
    ## #   pixel60 <int>, pixel61 <int>, pixel62 <int>, pixel63 <int>,
    ## #   pixel64 <int>, pixel65 <int>, pixel66 <int>, pixel67 <int>,
    ## #   pixel68 <int>, pixel69 <int>, pixel70 <int>, pixel71 <int>,
    ## #   pixel72 <int>, pixel73 <int>, pixel74 <int>, pixel75 <int>,
    ## #   pixel76 <int>, pixel77 <int>, pixel78 <int>, pixel79 <int>,
    ## #   pixel80 <int>, pixel81 <int>, pixel82 <int>, pixel83 <int>,
    ## #   pixel84 <int>, pixel85 <int>, pixel86 <int>, pixel87 <int>,
    ## #   pixel88 <int>, pixel89 <int>, pixel90 <int>, pixel91 <int>,
    ## #   pixel92 <int>, pixel93 <int>, pixel94 <int>, pixel95 <int>,
    ## #   pixel96 <int>, pixel97 <int>, pixel98 <int>, pixel99 <int>,
    ## #   pixel100 <int>, pixel101 <int>, pixel102 <int>, pixel103 <int>,
    ## #   pixel104 <int>, pixel105 <int>, pixel106 <int>, pixel107 <int>,
    ## #   pixel108 <int>, ...

``` r
# xgboost requirement: convert everything to numeric
train <- mutate_all(train, as.numeric)
test  <- mutate_all(test,  as.numeric)

# Separate training data and labels
X <- train %>% select(-label) %>% as.matrix()
y <- train$label

# Fit model
fit <- xgboost(X, y, nrounds = 25, objective = 'multi:softmax', num_class = 10)
```

    ## [1]  train-merror:0.126429 
    ## [2]  train-merror:0.085595 
    ## [3]  train-merror:0.070405 
    ## [4]  train-merror:0.060333 
    ## [5]  train-merror:0.054810 
    ## [6]  train-merror:0.049667 
    ## [7]  train-merror:0.044738 
    ## [8]  train-merror:0.039810 
    ## [9]  train-merror:0.035667 
    ## [10] train-merror:0.032405 
    ## [11] train-merror:0.028833 
    ## [12] train-merror:0.026238 
    ## [13] train-merror:0.023667 
    ## [14] train-merror:0.021071 
    ## [15] train-merror:0.019262 
    ## [16] train-merror:0.017238 
    ## [17] train-merror:0.015548 
    ## [18] train-merror:0.014024 
    ## [19] train-merror:0.012167 
    ## [20] train-merror:0.010643 
    ## [21] train-merror:0.009762 
    ## [22] train-merror:0.008929 
    ## [23] train-merror:0.008238 
    ## [24] train-merror:0.007381 
    ## [25] train-merror:0.006429

``` r
fit
```

    ## ##### xgb.Booster
    ## raw: 905.2 Kb 
    ## call:
    ##   xgb.train(params = params, data = dtrain, nrounds = nrounds, 
    ##     watchlist = watchlist, verbose = verbose, print_every_n = print_every_n, 
    ##     early_stopping_rounds = early_stopping_rounds, maximize = maximize, 
    ##     save_period = save_period, save_name = save_name, xgb_model = xgb_model, 
    ##     callbacks = callbacks, objective = "multi:softmax", num_class = 10)
    ## params (as set within xgb.train):
    ##   objective = "multi:softmax", num_class = "10", silent = "1"
    ## xgb.attributes:
    ##   niter
    ## callbacks:
    ##   cb.print.evaluation(period = print_every_n)
    ##   cb.evaluation.log()
    ##   cb.save.model(save_period = save_period, save_name = save_name)
    ## niter: 25
    ## evaluation_log:
    ##     iter train_merror
    ##        1     0.126429
    ##        2     0.085595
    ## ---                  
    ##       24     0.007381
    ##       25     0.006429

``` r
# Predict test data
predictions <- predict(fit, newdata = as.matrix(test)) %>% as.integer()

# Bundle into proper results table
results <- data.frame(
    ImageId = seq_len(nrow(test)),
    Label   = predictions
)

head(results)
```

    ##   ImageId Label
    ## 1       1     2
    ## 2       2     0
    ## 3       3     9
    ## 4       4     4
    ## 5       5     3
    ## 6       6     7

``` r
# Save results
write_csv(results, "../output/xgboost_benchmark.csv")
```
