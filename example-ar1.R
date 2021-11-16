rm(list=ls())
library(reshape2)
setwd("/Users/scabras74/gd/Articulos/elvirusdelacorona/covid19-bayes-dl/")
ccaa.lab=c("AN", "AR", "AS", "CB", "CE", "CL", "CM", "CN", "CT", 
           "EX", "GA","IB", "MC", "MD", "ML", "NC","PV", "RI", "VC")
cvirus.ccaa=read.table("https://cnecovid.isciii.es/covid19/resources/casos_diag_ccaadecl.csv", 
                          sep = ",", header = T)
maxr=max(which(cvirus.ccaa$fecha==max(cvirus.ccaa$fecha)))
cvirus.ccaa$ccaa_iso=factor(cvirus.ccaa$ccaa_iso)
cvirus.ccaa=data.frame(region=cvirus.ccaa$ccaa_iso,
                  day=as.Date(as.character(cvirus.ccaa$fecha),
                              format="%Y-%m-%d"),
                  cases=cvirus.ccaa$num_casos)
pop=read.table("demcca.csv",header = TRUE,sep=";")
pop=data.frame(region=pop$CCAA,pop=pop$pob)[1:19,]
pop=pop[order(pop$region),]
pop$w=pop$pop/sum(pop$pop)


cvirus.ccaa.fat=dcast(cvirus.ccaa,formula = day~region,value.var = "cases")
for(j in 2:ncol(cvirus.ccaa.fat)){
  y=cvirus.ccaa.fat[,j]
  y=c(rep(0,13),y)
  y=cumsum(y)
  y=y[14:length(y)]-y[1+(0:(length(y)-14))]
  cvirus.ccaa.fat[,j]=as.integer((y/pop$pop[pop$region==colnames(cvirus.ccaa.fat)[j]])*100000)
}

cvirus=cvirus.ccaa.fat
last.observed=max(cvirus$day)
dim(cvirus)

dp=cvirus$day[-1]
y=log(1+as.matrix(cvirus[,-1]))
yl1=y[-nrow(y),]
y=y[-1,]
start=7
ee=rr=array(NA,dim=dim(y))
res=res.errors=NULL
for(ndaysahead in 1:7){
for(i in start:(nrow(y)-ndaysahead)){
	mymod=lm(as.matrix(y[1:(i-1),])~as.matrix(yl1[1:(i-1),]))
	pred=round(exp(predict(mymod,newobs=yl1[1:(i+ndaysahead),])),0)
	rr[i+ndaysahead,]=pred[nrow(pred),]
	ee[i+ndaysahead,]=(exp(y[i+ndaysahead,])-1)-rr[i+ndaysahead,]
}
preds=data.frame(day=dp,rr)
colnames(preds)=colnames(cvirus)
preds$ndaysahead=ndaysahead
res=rbind(res,preds)
errors=data.frame(day=dp,ee)
colnames(errors)=colnames(cvirus)
errors$ndaysahead=ndaysahead
res.errors=rbind(res.errors, errors)

}

preds=res
errors=res.errors
obs=cvirus[-nrow(cvirus),]
obs$ndaysahead=0
preds$type="preds"
obs$type="obs"
errors$type="errors"
dd=rbind(preds,obs,errors)

dd=melt(dd,value.name="incidence",variable.name="region",id.vars=c("day","type","ndaysahead"))

library(ggplot2)
#ggplot(dd,aes(x=day,y=incidence,col=type))+facet_wrap(. ~ region)+
#geom_line()+
#theme(axis.text.x = element_text(angle = 90,size=rel(0.8)))

error=dd[dd$type=="errors",]

pdf(file="/Users/scabras74/gd/Articulos/elvirusdelacorona/mathematics/errorslm.pdf")
ggplot(error,aes(x=factor(ndaysahead),y= incidence))+
  facet_wrap(. ~ region)+geom_violin()+ ylab("Incidence")+ theme_bw()
dev.off()
