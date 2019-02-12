Metrics.data= matrix(0,nrow=length(file.list),ncol=1)
Metrics.data.2000.2017= matrix(0,nrow=length(file.list),ncol=1)
for (l in 1:length(file.list)){
	DF=read.xlsx(file.list[l],header=TRUE, sheetIndex=1) # read spreadsheet
	company.name=paste(sub("*...Ratios.*", "", colnames(DF)[1]),"METRICS")
	Drops<-c("1","2")
	DF=DF[!(rownames(DF) %in% Drops), ]
	for (i in 1:ncol(DF)){
		DF[,i]<-as.character(DF[,i])
		}
	rownames(DF) <- c("Ratios",as.character(unlist(DF[-1,1])))
	colnames(DF) <- as.character(unlist(DF[1,]))
	Drops<-c(NA,"Industry Median","Ratios")
	DF=DF[,!(colnames(DF) %in% Drops)]
	DF=DF[!(rownames(DF) %in% Drops),]
	Rowsrequired=c("Earnings Quality Score","Gross Margin","EBITDA Margin","Operating Margin","Pretax Margin",
	"Effective Tax Rate","Net Margin","Asset Turnover","Pretax ROA","x Leverage (Assets/Equity)",
	"Pretax ROE","x Tax Complement","ROE","x Earnings Retention","Reinvestment Rate","Quick Ratio","Current Ratio",
	"Times Interest Earned","Cash Cycle (Days)","Debt/Equity","% LT Debt to Total Capital","(Total Debt - Cash) / EBITDA")	
	DF=DF[Rowsrequired,]
	Rowsrequired2=c("Earnings Quality Score","Gross Margin","EBITDA Margin","Operating Margin","Pretax Margin",
	"Effective Tax Rate","Net Margin","Asset Turnover","Pretax ROA","Leverage (Assets/Equity)",
	"Pretax ROE","Tax Complement","ROE","Earnings Retention","Reinvestment Rate","Quick Ratio","Current Ratio",
	"Times Interest Earned","Cash Cycle (Days)","Debt/Equity","% LT Debt to Total Capital","(Total Debt - Cash) / EBITDA")	
	rownames(DF) <- Rowsrequired2
	DF[DF=="-"] <- NA
	DF[DF=="N/A"] <- NA
	DF=as.data.frame(DF)
	for (i in 1:ncol(DF)){
		DF[,i]<-as.numeric(as.character(DF[,i]))
		}
	DF=t(DF)
	Metrics.data[l,]=company.name
	assign(Metrics.data[l,],DF)
}
Metrics.data <- Metrics.data[ order(Metrics.data[,1]), ]
