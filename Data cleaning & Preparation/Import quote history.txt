setwd('C:/Users/soufiane/Desktop/Data_final')
### list all files in the working directory 
file.list<- list.files(pattern='*xls')
HQ.data= matrix(0,nrow=length(file.list),ncol=1)
for (l in 1:length(file.list)){
	DF=read.xlsx(file.list[l],header=TRUE, sheetIndex=1) # read spreadsheet
	company.name=paste(sub("*...Price.History.*", "", colnames(DF)[1]),"RET")
	NN=which(DF[,1] == "Exchange Date")
	Drops<-paste(1:NN-1)
	DF=DF[!(rownames(DF) %in% Drops), ]
	colnames(DF) <- as.character(unlist(DF[1,]))
	DF<-DF[-1,]
	Drops<-NA
	DF=DF[,!(colnames(DF) %in% Drops)]
	DFF=c(rep(1,nrow(DF)))
	for (i in 1:nrow(DF)){
		DFF[i]<-as.character(as.Date(as.numeric(as.character(DF[i,1])), origin="1899-12-30"))
		}
	DF$Date=DFF
	Rowsrequired=c("Date","Turnover","Approx VWAP","O-C","H-L","%CVol","%CTurn","Net","%Chg","Volume","Open","Low","High","Close")	
	DF=DF[,Rowsrequired]
	for (i in 2:ncol(DF)){
		DF[,i]<-as.numeric(as.character(DF[,i]))
		}
	DFxts <- xts(DF[,-1], order.by=as.Date(DF[,1], "%Y-%m-%d"))
	HQ.data[l,]=company.name
	assign(HQ.data[l,],DFxts)
}

