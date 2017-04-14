library(caret)

cvxTuneAndTrain = function(xtr, ytr, xtu, ytu, cvxcode, tuning.params, title, const.vars = list(), k = 10, delete.temp = FALSE, norun = FALSE, cvx.modifiers = "quiet", tempDir = "tmp"){
  # Setup folds
  tmp = createFolds(y = ytu, k = k)
  folds = rep(0, length(ytu))
  for(i in 1:length(names(tmp))){
    fold = tmp[[i]]
    for(j in 1:length(fold)){
      folds[fold[[j]]] = i
    }
  }
  
  infiles = c()
  outfiles = c()
  
  # Write relevant constant values to temp files
  constants = list(xtr = xtr, ytr = ytr, xtu = xtu, ytu = ytu, folds = folds)
  if(!missing(const.vars)){
    constants = append(constants, const.vars)
  }
  
  before <- ""
  if(!missing(title)){
    before <- sprintf("disp('Tuning %s parameters');", title)
  }
  
  for (i in seq(length(constants))) {
    input.name <- names(constants)[i]
    file <- sprintf("%s/in_%s.txt", tempDir, input.name)
    write.table(constants[[i]], file = file, row.names = FALSE, col.names = FALSE)
    before <- sprintf("%s%s = dlmread('%s');", before, input.name, file)
    infiles <- c(infiles, file)
  }
  
  # Write relevant tuning parameter values to temp files
  for (i in seq(length(tuning.params))) {
    input.name <- names(tuning.params)[i]
    file <- sprintf("%s/in_param_%s.txt", tempDir, input.name)
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
  
  before <- sprintf("%sresults = zeros(%d,%d);", before, resultsRows, length(param.names) + 1) # Last column is for MSE
  before <- sprintf("%sp = %d;", before, ncol(xtu))
  before <- sprintf("%sk = max(folds);", before)
  before <- sprintf("%scounter = 1;", before)
  before <- sprintf("%stStart=tic;", before)
  
  # Include a network norm anonymous function
  before <- sprintf("%snetnorm = @(x, netwk, wt, gamma) sum(norms([x(netwk(:,1))./wt(netwk(:,1)) x(netwk(:,2))./wt(netwk(:,2))], gamma, 2));", before)
  
  # Perform CV MSE calculation for all combinations of the tuning parameters
  crossValidationSnippet = sprintf("disp(sprintf('Combination %s of %s ( %s )', counter, (counter-1)*(100/%d))); cvMse = zeros(k,1); for fold = 1:k; training = find(folds ~= fold); holdout = find(folds == fold); x = xtu(training,:); y = ytu(training,:); cvx_begin %s; %s cvx_end; cvMse(fold) = mean((xtu(holdout,:)*b - ytu(holdout)).^2); end; mse = mean(cvMse);", 
                                   "%d", resultsRows, "%0.1f%%", resultsRows, cvx.modifiers, cvxcode)
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
  
  # Train the model with the best parameters from the tuning process
  core <- sprintf("%sdisp('Training model'); x = xtr; y = ytr; [minErrVal, minErrIdx] = min(results(:,%d));", core, length(param.names) + 1)
  for(i in 1:length(tuning.params)){
    core <- sprintf("%s%s=results(minErrIdx,%d);", core, param.names[[i]], i)
  }
  core <- sprintf("%scvx_begin %s; %s cvx_end;", core, cvx.modifiers, cvxcode)
  
  # Export the cross-validated results and time elapsed
  after <- "time=toc(tStart);"
  after <- sprintf("%sdisp('Done');", after)
  timeFile <- sprintf("%s/out_time.txt", tempDir)
  after <- sprintf("%sdlmwrite('%s', full(time), 'precision', '%s10.6f');", after, timeFile, "%")
  resultFile <- sprintf("%s/out_results.txt", tempDir)
  after <- sprintf("%sdlmwrite('%s', results, 'precision', '%s10.6f');", after, resultFile, "%")
  coefFile <- sprintf("%s/out_coefficients.txt", tempDir)
  after <- sprintf("%sdlmwrite('%s', b, 'precision', '%s10.6f');", after, coefFile, "%")
  paramFile <- sprintf("%s/out_parameters.txt", tempDir)
  after <- sprintf("%sdlmwrite('%s', results(minErrIdx,:), 'precision', '%s10.6f');", after, paramFile, "%")
  outfiles <- c(timeFile, resultFile, coefFile, paramFile)
  
  fileConn<-file("tmp/script.txt")
  writeLines(c(before, core, after, "exit;"), fileConn)
  close(fileConn)
  
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
  
  # Execute matlab script if required to do so
  if (norun) {
    return(command)
  }
  system(command)
  
  # Load results and prepare returned object
  output <- list()
  output$command <- command
  output$time <- drop(as.numeric(read.csv(timeFile, header = FALSE)))
  output$coefficients <- drop(as.matrix(read.csv(coefFile, header = FALSE)))
  output$tuning <- drop(as.matrix(read.csv(resultFile, header = FALSE)))
  output$params <- drop(as.matrix(read.csv(paramFile, header = FALSE)))
  output$fit <- structure(list(coefficients = output$coefficients), class = "RegFit")
  colnames(output$tuning) <- c(param.names, "MSE")
  names(output$params) <- c(param.names, "MSE")
  
  if (delete.temp) { 
    # Does not currently work due to unset Windows environment variables
    files <- c(infiles, outfiles)
    for(i in 1:length(files)){
      system(sprintf("%s %s", delCom, files[[i]]))
    }
  }
  
  return(output)
}

predict.RegFit = function(object, data, type = "response") {
  return(apply(data*object$coefficients,1,sum))
} 
