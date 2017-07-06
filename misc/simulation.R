library(MGLM)
data = read.table("data.csv")
write.table(as.data.frame(MGLMfit(data, dist="NegMN")$estimate), "res.csv", col.names = F, row.names = F)