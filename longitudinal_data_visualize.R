library(lmerTest)
# library(sjPlot)
library(glmmTMB)
library(ggplot2)
library(dplyr)
library(ggeffects)
library(effects)
library(readr)
library(ggChernoff)
library(merTools)
library(lattice)
library(stringr)
# library(tidyverse)
# ['moca','lns','hvlt','bjlot','sdmt','sft','pigd','tremor','seadl','updrs3','updrs2','updrs1']
if(dir.exists("/Volumes/Elements SE/ppmi_zheda/longitudinal_visualize/11.29_nopca") )
{   cat("dir exits","\n")
}else{dir.create("/Volumes/Elements SE/ppmi_zheda/longitudinal_visualize/11.29_nopca")}
setwd("/Volumes/Elements SE/ppmi_zheda/longitudinal_visualize/11.29_nopca")

data <- read.csv("/Volumes/Elements SE/ppmi_zheda/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_11.29/JS_updrs3_stack_all_no6.csv")#
data <- rename(data,subgroup=cluster)
dim(data) #1674   10

data <-subset(data,PATNO!=3116)
data <-subset(data,PATNO!=3771)
data <-subset(data,PATNO!=3818)
data <-subset(data,PATNO!=4114)
dim(data)
data$subgroup=factor(data$subgroup)
head(data)

png(file="u3_60Month_JS_no6.png",bg = "white",width=300*2,height=200*2,res=82*2)
ggplot(data=data,aes(x=month,y=values,group=subgroup,color=subgroup)) +
      geom_smooth(method = 'glm',se = TRUE,size=1,alpha=0.1,show.legend = FALSE)+
      # scale_color_tableau()+
      scale_color_manual(values =c("#dd8a0b","#32a676" ))+
      scale_fill_manual(values =c("grey100","grey100"))+
      scale_x_continuous( breaks = seq(0, 60, by = 12),  limits = c(0, 60))+
      scale_y_continuous( )+
      xlab("Time since diagnosis (months)")+
      # ylab(str_c("MDS-UPDRS ",as.character(as.roman("3"))))+
      ylab("MoCA")+
      guides(color=guide_legend(title="Subgroup"))+
      theme_classic()+
      theme(axis.title = element_text(size=9,  family="Times New Roman"),
            legend.title=element_text(size=9,  family="Times New Roman"),
            axis.line = element_line(size=0.4))




