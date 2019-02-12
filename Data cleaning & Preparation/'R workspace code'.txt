setwd('C:/Users/soufiane/Desktop/Data_final_metrics')
### list all files in the working directory 
file.list<- list.files(pattern='*xls')

### Keep only the data from 2000-2017
for (l in 1:length(Metrics.data)){	  
	DF=get(Metrics.data[l])
	Drops<-paste(1900:1999)
	DF=DF[!(rownames(DF) %in% Drops), ]
	assign(Metrics.data[l],DF)
	}


###Check for missing companies in RET data
NAcompRET=""
for (i in 1:(length(HQ.data))) {
	Tester=sub("* RET*", "", HQ.data[i])
	Cond=FALSE
	for (j in 1:(length(Metrics.data))){
	if (sapply(Tester, grepl, Metrics.data[j])==TRUE){
	Cond=TRUE
	}
	}
	if (Cond==FALSE){
	NAcompRET=c(NAcompRET,HQ.data[i])
	}
	}
###Check for missing companies in METRICS data
NAcompMETRICS=""
for (i in 1:(length(Metrics.data))) {
	Tester=sub("* METRICS*", "", Metrics.data[i])
	Cond=FALSE
	for (j in 1:(length(HQ.data))){
	if (sapply(Tester, grepl, HQ.data[j])==TRUE){
	Cond=TRUE
	}
	}
	if (Cond==FALSE){
	NAcompMETRICS=c(NAcompMETRICS,Metrics.data[i])
	}
	}
	
HQ.data=HQ.data[!(HQ.data %in% NAcompRET[-1])]
Metrics.data=Metrics.data[!(Metrics.data %in% NAcompMETRICS[-1])]
### recheck vice versa order
Metrics.data <- Metrics.data[ order(Metrics.data)]
HQ.data <- HQ.data[ order(HQ.data)]
for (i in 1:(length(Metrics.data))) {
	Tester=sub("* METRICS*", "", Metrics.data[i])
	print(sapply(Tester, grepl, HQ.data[i]))
	}

### Risk free rate
RFR=read.xlsx('C:/Users/soufiane/Desktop/US10Y.xlsx',header=TRUE, sheetIndex=1) # read spreadsheet
keeps<-c("Date","Price")
RFR=RFR[,keeps]
RFR$Date<- mdy(RFR$Date)
RFRxts <- xts(RFR[,-1], order.by=as.Date(RFR[,1], "%Y-%m-%d"))


### Benchmarking rate
BR=read.xlsx('C:/Users/soufiane/Desktop/SP500.xlsx',header=TRUE, sheetIndex=1) # read spreadsheet
keeps<-c("Date","Close")
BR=BR[,keeps]
BRxts <- xts(BR[,-1], order.by=as.Date(BR[,1], "%Y-%m-%d"))

####create yearly database of monthlyReturn  
Monthly.return.data.2000.2017= matrix(0,nrow=(2017-2000+1),ncol=1)
rownames(Monthly.return.data.2000.2017)=paste(2000:2017)
Monthly.return.data.2000.2017[,1]=paste("MRET_",2000:2017)

for (k in 1:nrow(Monthly.return.data.2000.2017)){
	if (rownames(Monthly.return.data.2000.2017)[k]==2000) {
	Monthly.return.data.template= matrix(0,nrow=length(HQ.data),ncol=15)
	rownames(Monthly.return.data.template)=HQ.data
	colnames(Monthly.return.data.template)=c('02','03','04','05','06','07','08','09','10','11','12','Mean','Variance','Skewness','Kurtosis')
	} else {
	Monthly.return.data.template= matrix(0,nrow=length(HQ.data),ncol=16)
	rownames(Monthly.return.data.template)=HQ.data
	colnames(Monthly.return.data.template)=c('01','02','03','04','05','06','07','08','09','10','11','12','Mean','Variance','Skewness','Kurtosis')
	}
	for (j in 1:nrow(Monthly.return.data.template)) {
		DS=get(rownames(Monthly.return.data.template)[j])
		DS=DS[,"Close"]
		DSmonthlyReturn<-monthlyReturn(DS)
		for (i in 1:(ncol(Monthly.return.data.template)-4)){
			Cond=paste0(rownames(Monthly.return.data.2000.2017)[k],colnames(Monthly.return.data.template)[i])
			if (is.null(DSmonthlyReturn[Cond,which.i=TRUE])==FALSE){
			Monthly.return.data.template[j,i]=DSmonthlyReturn[Cond]
			}else{Monthly.return.data.template[j,i]=NA}
			}
			
			Cond.year=rownames(Monthly.return.data.2000.2017)[k]
			if (is.null(DSmonthlyReturn[Cond.year,which.i=TRUE])==FALSE){
			Monthly.return.data.template[j,'Mean']=mean(DSmonthlyReturn[Cond.year], na.rm = TRUE)
			Monthly.return.data.template[j,'Variance']=var(DSmonthlyReturn[Cond.year], na.rm = TRUE)
			Monthly.return.data.template[j,'Skewness']=skewness(DSmonthlyReturn[Cond.year], na.rm = TRUE)
			Monthly.return.data.template[j,'Kurtosis']=kurtosis(DSmonthlyReturn[Cond.year], na.rm = TRUE)
			}else{
			Monthly.return.data.template[j,'Mean']=NA
			Monthly.return.data.template[j,'Variance']=NA
			Monthly.return.data.template[j,'Skewness']=NA
			Monthly.return.data.template[j,'Kurtosis']=NA
			}
		}
	
	assign(Monthly.return.data.2000.2017[k,1],Monthly.return.data.template)
	}

#### create a dataset of METRICS
METRICS.data.2000.2017= matrix(0,nrow=(2017-2000+1),ncol=1)
rownames(METRICS.data.2000.2017)=paste(2000:2017)
METRICS.data.2000.2017[,1]=paste("METRICS_",2000:2017)
METRICS.data.template= matrix(NA,nrow=length(Metrics.data),ncol=ncol(get(Metrics.data[1])))
rownames(METRICS.data.template)=Metrics.data
colnames(METRICS.data.template)=colnames(get(Metrics.data[1]))

for (k in 1:nrow(METRICS.data.2000.2017)){
	for (j in 1:nrow(METRICS.data.template)) {
		DS=get(rownames(METRICS.data.template)[j])
		for (i in 1:nrow(DS)){
			Cond=isTRUE(rownames(DS)[i]==rownames(METRICS.data.2000.2017)[k])
			if (Cond==TRUE) {
			METRICS.data.template[j,]=DS[i,]
			} 
			
			}
			
		}
	assign(METRICS.data.2000.2017[k,1],METRICS.data.template)
	}

### Consolidate final database
Final.consolidated.data.2000.2017= matrix(0,nrow=(2017-2000+1),ncol=1)
rownames(Final.consolidated.data.2000.2017)=paste(2000:2017)
Final.consolidated.data.2000.2017[,1]=paste("MRET_METRICS_",2000:2017)
for (l in 1:length(Final.consolidated.data.2000.2017)){	 
assign(Final.consolidated.data.2000.2017[l,1],cbind(get(Monthly.return.data.2000.2017[l,]),get(METRICS.data.2000.2017[l,])))
}