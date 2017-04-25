csvExportRData = function(filename){
  variableName = load(paste(filename,".RData", sep = ""))
  write.csv(get(variableName), file = paste(filename,".csv", sep = ""))
}

csvExportRData("tumor_data/adjm_body_data")
csvExportRData("tumor_data/adjm_prom_data")
csvExportRData("tumor_data/meth_body_data")
csvExportRData("tumor_data/meth_prom_data")
