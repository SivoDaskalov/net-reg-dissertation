csvExportRData = function(filename){
  variableNames <<- load(paste(filename,".RData", sep = ""))
  for (i in 1:length(variableNames)){
    print(variableNames[i])
    write.csv(get(variableNames[i]), file = paste(filename, "_", variableNames[i],  ".csv", sep = ""))
  }
}

csvExportRData("tumor_data/adjm_body_data")
csvExportRData("tumor_data/adjm_prom_data")
csvExportRData("tumor_data/meth_body_data")
csvExportRData("tumor_data/meth_prom_data")