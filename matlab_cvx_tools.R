# library(CVXfromR)
library(caret)

n <- 50
p <- 20
k <- 10
x <- matrix(rnorm(n * p), n, p)
y <- x %*% rnorm(p) + 0.1 * rnorm(n)
tuning.params = list(gam = c(1,2,3), lam = c(4,5,6))
cvxcode <- paste("variables b(p)",
                 "minimize(square_pos(norm(y - x * b, 2)) / 2 + lam * norm(b, 1) / gam);",
                 sep="; ")
cvx.modifiers = "quiet"
# const.vars = list(k = k, n = n, p = p)
# delete.temp = TRUE
cvxResults = CallCvx.CV(x = x, y = y, cvxcode = cvxcode, tuning.params = tuning.params, k = 10)
errors = cvxResults$results

CallCvx.CV = function(x, y, cvxcode, const.vars = list(), tuning.params, k, delete.temp = TRUE, norun = FALSE, cvx.modifiers = "quiet"){
  
  # Setup folds
  tmp = createFolds(y = y, k = k)
  folds = rep(0, length(y))
  for(i in 1:length(names(tmp))){
    fold = tmp[[i]]
    for(j in 1:length(fold)){
      folds[fold[[j]]] = i
    }
  }
  
  infiles = c()
  outfiles = c()
   
  # Write relevant constant values to temp files
  constants = list(x = x, y = y, folds = folds)
  # if(!missing(const.vars)){
  #   constants = append(constants, const.vars)
  # }
  
  before = ""
  for (i in seq(length(constants))) {
    input.name <- names(constants)[i]
    file <- sprintf("tmp/in_%s.txt", input.name)
    write.table(constants[[i]], file = file, row.names = FALSE, col.names = FALSE)
    before <- sprintf("%s%s = dlmread('%s');", before, input.name, file)
    infiles <- c(infiles, file)
  }
  
  # Write relevant tuning parameter values to temp files
  for (i in seq(length(tuning.params))) {
    input.name <- names(tuning.params)[i]
    file <- sprintf("tmp/in_param_%s.txt", input.name)
    write.table(tuning.params[[i]], file = file, row.names = FALSE, col.names = FALSE)
    before <- sprintf("%s%s_all = dlmread('%s');", before, input.name, file)
    infiles <- c(infiles, file)
  }
  
  # Initialize various matlab values
  resultsRows = 1
  param.names = names(tuning.params)
  for(i in 1:length(param.names)){
    resultsRows = resultsRows * length(tuning.params[[i]])
  }
  before <- sprintf("%sorigX = x;", before)
  before <- sprintf("%sorigY = y;", before)
  before <- sprintf("%sresults = zeros(%d,%d);", before, resultsRows, length(param.names) + 1) # Last column is for MSE
  before <- sprintf("%sp = %d;", before, ncol(x))
  before <- sprintf("%sk = max(folds);", before)
  before <- sprintf("%scounter = 1;", before)
  before <- sprintf("%stStart=tic;", before)
  
  # Perform CV MSE calculation for all combinations of the tuning parameters
  crossValidationSnippet = sprintf("cvMse = zeros(k,1); for fold = 1:k; training = find(folds ~= fold); holdout = find(folds == fold); x = origX(training,:); y = origY(training,:); cvx_begin %s; %s cvx_end; cvMse(fold) = mean((origX(holdout,:)*b - origY(holdout)).^2); end; mse = mean(cvMse);", cvx.modifiers, cvxcode)
  core = ""
  for(i in 1:length(tuning.params)){
    core <- sprintf("%sfor %s_it=1:size(%s_all); %s=%s_all(%s_it);", core, 
            param.names[[i]], param.names[[i]], param.names[[i]], param.names[[i]], param.names[[i]])
  }
  core <- sprintf("%s %s results(counter,:) = [%s,mse]; counter = counter + 1;", 
                  core, crossValidationSnippet, paste(param.names, collapse = ","))
  for(i in 1:length(tuning.params)){
    core <- sprintf("%send;", core)
  }
  
  # Export the cross-validated results and time elapsed
  after <- "time=toc(tStart);"
  timeFile <- "tmp/out_time.txt"
  after <- sprintf("%sdlmwrite('%s', full(time), 'precision', '%s10.10f');", after, timeFile, "%")
  resultFile <- "tmp/out_results.txt"
  after <- sprintf("%sdlmwrite('%s', results, 'precision', '%s10.10f');", after, resultFile, "%")
  outfiles <- c(timeFile, resultFile)
    
  # Bind together generated matlab code chunks with respect to the operating system
  if (.Platform$OS.type == "unix") {
    command <- sprintf("matlab -nodisplay -r \"%s%s%s\"", before, core, after, "exit;")
    delCom = "del"
  }  else if (.Platform$OS.type == "windows") {
    command <- sprintf("matlab -wait -nosplash -nodesktop -r \"%s%s%s%s\"", before, core, after, "exit;")
    delCom = "del"
  }  else {
    stop("Not a recognized operating system.")
  }
  core
  command
  # Execute matlab script if required to do so
  if (norun) {
    return(command)
  }
  system(command)
  
  # Load results and prepare returned object
  output <- list()
  output$command <- command
  output$time <- drop(as.numeric(read.csv(timeFile, header = FALSE)))
  output$results <- drop(as.matrix(read.csv(resultFile, header = FALSE)))
  colnames(output$results) <- c(param.names, "MSE")
  
  # if (delete.temp) {
  #   files <- c(infiles, outfiles)
  #   for(i in 1:length(files)){
  #     system(sprintf("%s %s", delCom, files[[i]]))
  #   }
  # }
  
  return(output)
}
