library(lmerTest)
# library(sjPlot)
library(glmmTMB)
library(ggplot2)
library(dplyr)
library(ggeffects)  #ayestestR, parameters, pROC, sjstats, epiDisplay
library(effects)
library(readr)
library(ggChernoff)
library(merTools)
library(lattice)
# library(epiR)
# library(epicalc)
# library(pROC)
# library(lattice)
library(epiDisplay)
library(lm.beta)
# library(tidyverse)
# ['moca','lns','hvlt','bjlot','sdmt','sft','pigd','tremor','seadl','updrs3','updrs2','updrs1']
if(dir.exists("/Volumes/Elements SE/ppmi_zheda/linear_mixed_effect_model_slop/11.29_nopca") )
{   cat("dir exits","\n")
}else{dir.create("/Volumes/Elements SE/ppmi_zheda/linear_mixed_effect_model_slop/11.29_nopca")}
setwd("/Volumes/Elements SE/ppmi_zheda/linear_mixed_effect_model_slop/11.29_nopca")

data <- read.csv("/Volumes/Elements SE/ppmi_zheda/step4_result_739/latent_vecs100_cvae_bspline_10_1_0.9_0.001/cluster_result_kmeans_11.29/nopca_follow_varia_stack_subgroup/moca_stack_all.csv")#
data <- rename(data,subgroup=cluster)
# dim(data) #1674   10
# head(data)
data$cluster=factor(data$subgroup)
data$SITE=factor(data$SITE)
data$gen=factor(data$gen)
data$race=factor(data$race)
#pca

# pca1 <- prcomp(iris_input[,-ncol(iris_input)],center = TRUE,scale. = TRUE)
# df1 <- pca1$x # 提取PC score
# df1 <- as.data.frame(df1) # 注意：如果不转成数据框形式后续绘图时会报错
# summ1 <- summary(pca1) #Importance of components:
# summ1 <- summary(pca1)
# xlab1 <- paste0("PC1(",round(summ1$importance[2,1]*100,2),"%)")
# ylab1 <- paste0("PC2(",round(summ1$importance[2,2]*100,2),"%)")
# library(ggplot2)
#
# p.pca1 <- ggplot(data = df1,aes(x = PC1,y = PC2,color = iris_input$Species))+
#   stat_ellipse(aes(fill = iris_input$Species),
#                type = "norm",geom = "polygon",alpha = 0.25,color = NA)+ # 添加置信椭圆
#   geom_point(size = 3.5)+
#   labs(x = xlab1,y = ylab1,color = "Condition",title = "PCA Scores Plot")+
#   guides(fill = "none")+
#   theme_bw()+
#   scale_fill_manual(values = c("purple","orange","pink"))+
#   scale_colour_manual(values = c("purple","orange","pink"))+
#   theme(plot.title = element_text(hjust = 0.5,size = 15),
#         axis.text = element_text(size = 11),axis.title = element_text(size = 13),
#         legend.text = element_text(size = 11),legend.title = element_text(size = 13),
#         plot.margin = unit(c(0.4,0.4,0.4,0.4),'cm'))
# ggsave(p.pca1,filename = "PCA.pdf")
# latent=df$PC1
data.fullmodel_1 = lmer(values ~ age + gen + race + SITE+ EDUCYRS+ subgroup*month +(1 + month|PATNO),
data=data,REML=FALSE)
# data.fullmodel_1 = lmer(values ~ age + gen + race + EDUCYRS+ month +(1 + month|PATNO),
# data=data,REML=FALSE)

# data.fullmodel_2 = lmer(values ~ age + gen + race + EDUCYRS+ subgroup*month +(1 |PATNO)+ (1 |subgroup),
# data=data,REML=FALSE)
# data.fullmodel_3 = lmer(values ~  subgroup*month +(1 |PATNO)+ (1|subgroup),
# data=data,REML=FALSE)#age + gen + race + EDUCYRS+
# data.zore=lmer(values ~ age + gen + race + EDUCYRS+ subgroup*month +(1|PATNO),data=data,REML=FALSE)
# data.max=lmer(values ~ age + gen + race + EDUCYRS+ subgroup*month +(subgroup*month|PATNO),data=data,REML=FALSE)
# data.model2=lmer(values ~ age + gen + race + EDUCYRS+ subgroup*month +(1+month|PATNO),data=data,REML=FALSE)
# anova(data.zore)
# anova(data.max)

summary(data.fullmodel_1)

stdCoef.lmer <- function(object) {
  sdy <- sd(attr(object, "y"))
  sdx <- apply(attr(object, "X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  #mimic se.ranef from pacakge "arm"
  se.fixef <- function(obj) attr(summary(obj), "coefs")[,2]
  se <- se.fixef(object)*sdx/sdy
  return(list(stdcoef=sc, stdse=se))
}
stdCoef.merMod <- function(object) {
  sdy <- sd(getME(object,"y"))
  sdx <- apply(getME(object,"X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  se.fixef <- coef(summary(object))[,"Std. Error"]
  se <- se.fixef*sdx/sdy
  return(data.frame(stdcoef=sc, stdse=se))
}
stdCoef.merMod(data.fullmodel_1)
sigma(data.fullmodel_1)
# confint(data.fullmodel_1)
# tab(data.fullmodel_1)
# coef(data.fullmodel_1)
fixef(data.fullmodel_1)
# coef_lmbeta <- lm.beta(data.fullmodel_1)
# anova(data.fullmodel_1)
# isSingular(data.fullmodel_1) #TRUE 奇异拟合了

# coef.data.fullmodel_1 <- summary(data.fullmodel_1)$tTable
# # print(coef.data.fullmodel_1,vcov(x))
# vcov(coef.data.fullmodel_1)
# ci(coef.data.fullmodel_1[,1],coef.data.fullmodel_1[,2],coef.data.fullmodel_1[,4])



days_se <- sqrt(diag(vcov(data.fullmodel_1)))[34]
days_coef <- fixef(data.fullmodel_1)[34]
cat("fixef(data.fullmodel_1)[34]",fixef(data.fullmodel_1)[34])
upperCI <-  days_coef + 1.96*days_se
lowerCI <-  days_coef  - 1.96*days_se
cat("upperCI",upperCI,"lowerCI",lowerCI)


#
# fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
# summary(fm1)
# fixef(fm1)
# stdCoef.merMod(fm1)
#
# days_se_fm1 <- sqrt(diag(vcov(fm1)))[2]
# days_coef_fm1 <- fixef(fm1)[2]
# cat("days_coef_fm1",fixef(fm1)[1])
#
# upperCI <-  days_coef_fm1 + 1.96*days_se_fm1
# lowerCI <-  days_coef_fm1  - 1.96*days_se_fm1
# cat("upperCI",upperCI,"lowerCI",lowerCI)

# effects_zero=effect(term=("month"),mod=data.fullmodel_1)
# summary(effects_zero)
# x_zero <- as.data.frame(effects_zero)
# head(x_zero)
# print("fixef")
# print(lme4::fixef(data.fullmodel_1))##输出固定效应系数
# fiexd=as.data.frame(lme4::fixef(data.fullmodel_1))
# print(dim(fiexd)) #10  1
# print(fiexd)
# print("ranef")
# sloper=as.data.frame(lme4::ranef(data.fullmodel_1)$PATNO[,2])
# # write_csv(sloper, "updrs1_stack_all_sloper.csv")
# sloper_2=as.data.frame(lme4::ranef(data.fullmodel_1))
# print(lme4::ranef(data.fullmodel_1))#输出随机效应系数
# print("slop")
# print(sloper)
# print(dim(sloper)) #321   1
# print(dim(sloper_2)) # 642   5
# print(head(sloper_2))
# write_csv(sloper_2, "updrs1_stack_all_interp_sloper.csv")



# coef(data.fullmodel_1)#intercept
# confint(data.fullmodel_1, level = 0.9)#查看随机效应标准差和固定效应系数的置信区间
# # isSingular(data.fullmodel_2)qqmath(ranef(fit1, postVar = TRUE), strip = FALSE)$Batch
# pdf(file="qqmath.pdf",height=3,width=4)
# qqmath(ranef(data.fullmodel_1, condVar = TRUE), strip = FALSE)$PATNO
# dev.off()

# ggCaterpillar(ranef(fit, condVar=TRUE))

# randoms <- REsim(data.fullmodel_1, n.sims = 500)
# pdf(file="plotREsim.pdf",height=3,width=4)
# plotREsim(randoms)
# dev.off()
# anova(data.fullmodel_1, data.fullmodel_2)
# summary(data.fullmodel_2)
# summary(data.fullmodel_3)
# isSingular(data.fullmodel_3)
# model_max =lmer(data = data,formula = )

# summary(data.zore)
# isSingular(data.zore)
# summary(data.max)
# isSingular(data.max)
# anova(data.zore,data.max)
# summary(data.model2)
# anova(data.model2)


# coef(data.model2)
# pdf(file="ranef.pdf",height=3,width=4)
# plot(ranef(data.zore))
# dev.off()
# pdf(file="residual.pdf",height=3,width=4)
# plot(data.zore)
# dev.off()
# pdf(file="model2.pdf",height=3,width=4)
# plot_model(data.zore, type = "re", show.values = TRUE)
# dev.off()
#查看每个ID分别计算的截距和斜率
# coefficients(data.zore)

# library(ggeffects)
# pdf(file="model_fit_moca.pdf",height=3,width=4)
# pred.mm <- ggpredict(data.zore, terms =c("month","subgroup"),se=TRUE,interactive=TRUE)
# plot(pred.mm)
# dev.off()


# #
# pdf(file="model_fit2.pdf",height=3,width=4)
# pred.mm <- ggpredict(data.zore, terms =c("month","cluster"),se=TRUE,interactive=TRUE)
# ggplot(pred.mm,group=cluster) +
#   #geom_point(data = data,aes(x = month, y = values, colour = cluster),position = "jitter")
#   geom_line(aes(x = x, y = predicted)) +          # slope
#   geom_ribbon(aes(x = x, ymin = predicted - std.error, ymax = predicted + std.error),
#               fill = "lightgrey", alpha = 0.5) + # error band
#   theme_minimal()
# dev.off()



