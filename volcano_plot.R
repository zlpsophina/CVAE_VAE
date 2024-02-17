## Figure 1  Overall results volcano plot```{r}


library(ggplot2)
library(ggrepel)
require(dplyr)
require(tidyr)
require(RColorBrewer)

rm(list=ls())

setwd('/Volumes/Elements SE/ppmi_zheda/kegg')

# corr_proteins <- read.csv('/Volumes/Elements SE/ppmi_zheda/step3_RSA_739/latent_vecs10_cvae_bspline_10_1_0.9_0.001/T_tabel_result_csf_fdr_bh_raw_p_2.csv')
corr_proteins <- read.csv('/Volumes/Elements SE/ppmi_zheda/step3_RSA_739/latent_vecs10_cvae_bspline_10_1_0.9_0.001/csf_proteins_v2/T_tabel_result_csf_fdr_bh_5variables_ols.csv')


corr_proteins$bonferroni_p <- p.adjust(corr_proteins$raw_p,method="bonferroni",n=length(corr_proteins$raw_p))
corr_proteins$FDR_p <- p.adjust(corr_proteins$raw_p,method="BH",n=length(corr_proteins$raw_p))

corr_proteins$log_p_value <- -log10(corr_proteins$raw_p)

head(corr_proteins)
a<-sum(corr_proteins$bonferroni_p<0.05)
b<-sum(corr_proteins$FDR_p<0.01)
print(a)
print(b)

# corr_proteins$group <- lapply(corr_proteins$bonferroni_p, function(x) {
#
#   y <- ifelse(x < 0.01, 1, 0)
#   return(y)
# })


corr_proteins$group <- lapply(corr_proteins$FDR_p, function(x) {

  y <- ifelse(x < 0.05, 1, 0)
  return(y)
})




corr_proteins$color <- lapply(corr_proteins$FDR_p, function(x) {

  y <- ifelse(x < 0.05, "red", "grey")
  return(y)
})




corr_proteins$opaque <- lapply(corr_proteins$FDR_p, function(x) {

  y <- ifelse(x < 0.05, 1, 0.5)
  return(y)
})

# write.csv("T_tabel_result_csf_fdr_bh_raw_p_2_log_5varibales_ols.csv",row.names = F)

# head(corr_proteins)

thr1=-log10(0.05/length(corr_proteins$raw_p))

thr2=-log10(max(corr_proteins$raw_p[corr_proteins$FDR_p<0.05]))




class(corr_proteins$group)


p<-ggplot(corr_proteins, aes(x=raw_t , y=log_p_value)) +
  # geom_hline(yintercept=thr1, color="grey", linewidth=0.4,linetype="longdash")+
  geom_hline(yintercept=thr2, color="grey", size=0.4,linetype="longdash")+
  geom_point(aes(colour = color),alpha=0.4, size=1) +
  #shape=Correlation_directions
  labs(color="group", x="T values", y=expression('-Log10(p-value)'),) +
  ggrepel::geom_label_repel(data=. %>% mutate(label = ifelse(corr_proteins$raw_p < 0.00016, as.character(key), "")), aes(label=label),
                  size=2 , box.padding = unit(0.2, "lines"),#bonferroni_pFDR_p< 0.01
                  max.overlaps = getOption("ggrepel.max.overlaps", default = 1800),
                 # # seed = 233,
                   color = "#0072b5ff",
                  force = 2,
                  force_pull = 2,
                  min.segment.length = 0,

  ) +
  theme_classic() + 
  theme(axis.title = element_text(size=9,family="Times New Roman"),
        axis.text = element_text(size = 7,family="Times New Roman"),
        
        axis.line.y = element_line(color = "black",linewidth=0.2),
        axis.line.x = element_line(color = "black",linewidth=0.2),
        
        #axis.ticks.x = element_blank(),
        
        axis.text.x = element_text(color = "black",size=7),
        axis.text.y = element_text(color = "black",size=7),
  )
ggsave('volcanoplot_Rvalue_figure3__5varibales_ols_fdr0.01.png', p, dpi = 300, width = 100, height = 100, units = "mm")


