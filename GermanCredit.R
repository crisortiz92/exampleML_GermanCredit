#Machine Learning USe CAse: German Credit DATA

  library("data.table")
  library("mlr3")
  library("mlr3learners")
  library("mlr3viz")
  library("ggplot2")

#The goal is to classify people by their credit risk (good or bad) 
#using 20 personal, demographic and financial features:

  #Importing data
  
  data("german", package = "rchallenge")
  
  #Exploring the data
  
  dim(german)
  str(german)

#Using skimr and DataExplorer as they create very well readable and
#understandable overviews:
  
  skimr::skim(german)
  
#Prior to calling DataExplorer, we shorten the (very lengthy) factor 
#levels of the german credit data set to get plots which nicely fit 
#on the screen.
  
  german_short = german
  is_factor = sapply(german_short, is.factor)
  german_short[is_factor] = lapply(german[is_factor], function(x){
    levels(x) = abbreviate(mlr3misc::str_trunc(levels(x), 16, "..."), 12)
    x
  })
  
  #Normal View
  DataExplorer::plot_bar(german, nrow = 6, ncol = 3)
  
  #Improve text format 
  DataExplorer::plot_bar(german_short, nrow = 6, ncol = 3)
  
  DataExplorer::plot_histogram(german_short, nrow = 1, ncol = 3)
  
  DataExplorer::plot_boxplot(german_short, by = "credit_risk", nrow = 1, ncol = 3)
  
  
#he typical questions that arise when building a machine learning workflow are:
  
  # What is the problem we are trying to solve?
  #   What are appropriate learning algorithms?
  #   How do we evaluate "good" performance?
  #   
  #   More systematically in mlr3 they can be expressed via five components:
  #   
  #   The Task definition.
  # The Learner definition.
  # The training.
  # The prediction.
  # The evaluation via one or multiple Measures.
  # 
  
  
  task = TaskClassif$new("GermanCredit", german, target = "credit_risk")
  
  #Using Logistic Regresion
  
  library("mlr3learners")
  learner_logreg = lrn("classif.log_reg")
  print(learner_logreg)
  
  #Training
  
  learner_logreg$train(task)
  train_set = sample(task$row_ids, 0.8 * task$nrow)
  test_set = setdiff(task$row_ids, train_set)
  
  head(train_set)
  
  learner_logreg$train(task, row_ids = train_set)
  
  #The fitted model can be accessed via:
  
  learner_logreg$model
  
  class(learner_logreg$model)

  summary(learner_logreg$model)  
  
  
  #Using random forest
  
  learner_rf = lrn("classif.ranger", importance = "permutation")
  learner_rf$train(task, row_ids = train_set)
  learner_rf$importance()
  
  #Importance in plot ggplot2
  
  importance = as.data.table(learner_rf$importance(), keep.rownames = TRUE)
  colnames(importance) = c("Feature", "Importance")
  ggplot(importance, aes(x = reorder(Feature, Importance), y = Importance)) + 
    geom_col() + coord_flip() + xlab("")
  
  #PRediction process
  
  pred_logreg = learner_logreg$predict(task, row_ids = test_set)
  pred_rf = learner_rf$predict(task, row_ids = test_set)
  
  pred_logreg
  pred_rf
  
  #confusion matrix
  
  pred_logreg$confusion
  pred_rf$confusion
  
  #Add probability scale
  
  learner_logreg$predict_type = "prob"
  learner_logreg$predict(task, row_ids = test_set)
  
  resampling = rsmp("holdout", ratio = 2/3)
  print(resampling)
  
  res = resample(task, learner = learner_logreg, resampling = resampling)
  res
  
  res$aggregate()
  
  resampling = rsmp("subsampling", repeats = 10)
  rr = resample(task, learner = learner_logreg, resampling = resampling)
  rr$aggregate()
  
  resampling = resampling = rsmp("cv", folds = 10)
  rr = resample(task, learner = learner_logreg, resampling = resampling)
  rr$aggregate()
  
  # false positive rate
  rr$aggregate(msr("classif.fpr"))
  
  # false positive rate and false negative
  measures = msrs(c("classif.fpr", "classif.fnr"))
  rr$aggregate(measures)
  
  mlr_resamplings
  mlr_measures
  
  learners = lrns(c("classif.log_reg", "classif.ranger"), predict_type = "prob")
  bm_design = benchmark_grid(
    tasks = task,
    learners = learners,
    resamplings = rsmp("cv", folds = 10)
  )
  bmr = benchmark(bm_design)
  
  measures = msrs(c("classif.ce", "classif.auc"))
  performances = bmr$aggregate(measures)
  performances[, c("learner_id", "classif.ce", "classif.auc")]
  
  learner_rf$param_set
  
  learner_rf$param_set$values = list(verbose = FALSE)
  
  ## ?ranger::ranger
  as.data.table(learner_rf$param_set)[, .(id, class, lower, upper)]
  
  #Performance
  rf_med = lrn("classif.ranger", id = "med", predict_type = "prob")
  
  rf_low = lrn("classif.ranger", id = "low", predict_type = "prob",
               num.trees = 5, mtry = 2)
  
  rf_high = lrn("classif.ranger", id = "high", predict_type = "prob",
                num.trees = 1000, mtry = 11)
  
  learners = list(rf_low, rf_med, rf_high)
  bm_design = benchmark_grid(
    tasks = task,
    learners = learners,
    resamplings = rsmp("cv", folds = 10)
  )
  
  
  bmr = benchmark(bm_design)
  print(bmr)
  
  measures = msrs(c("classif.ce", "classif.auc"))
  performances = bmr$aggregate(measures)
  performances[, .(learner_id, classif.ce, classif.auc)]
  
  autoplot(bmr)
  
  # 
  # The "low" settings seem to underfit a bit, the "high" setting
  # is comparable to the default setting "med".
  # 
  
  #Reference:
  # Binder, et al. (2020, March 11). mlr3gallery: mlr3 Basics - German Credit. 
  # Retrieved from https://mlr3gallery.mlr-org.com/posts/2020-03-11-basics-german-credit/
  
  
  
  
  
  
  
  
  
  
  
  
  
  