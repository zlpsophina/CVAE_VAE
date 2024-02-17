library(ggplot2)
library(dplyr)
library(ggplotAssist)
library(DOSE)
library(enrichplot)
library(clusterProfiler)
library(org.Hs.eg.db)
library(gridExtra)
setwd("/Volumes/Elements SE/ppmi_zheda/kegg")
# data(geneList)
keytypes(org.Hs.eg.db)
gene_data <- read.csv("/Volumes/Elements SE/PPMI/csf_proteins_fdr_bh_0.05_key_5varibales_ols.csv")
all_gene <- read.csv("/Volumes/Elements SE/PPMI/all_key.csv")
# all_gene <- read.csv("/Volumes/Elements SE/ppmi_zheda/step3_RSA_739/latent_vecs10_cvae_bspline_10_1_0.9_0.001/csf_proteins_v2/T_tabel_result_csf_fdr_bh.csv")
gene <- gene_data$key #converted_alias
all_gene_list <-all_gene$key
# head(all_gene_list)
gene <- bitr(gene,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = "org.Hs.eg.db",drop = T)
all_gene_list <- bitr(all_gene_list,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = "org.Hs.eg.db",drop = T)

# head(gene)

all_BP <- enrichGO(gene = gene$ENTREZID,
               OrgDb = org.Hs.eg.db,
               keyType='ENTREZID',
               pAdjustMethod = "BH",
               pvalueCutoff =0.05,
               qvalueCutoff = 0.05,
               ont="BP", #"MF", "CC",
               readable =T)
dim(all_BP)
all_BP_simp <- simplify(all_BP, cutoff=0.7, by="p.adjust", select_fun=min)
dim(all_BP_simp)
kk <- enrichGO(gene = gene$ENTREZID,
               OrgDb = org.Hs.eg.db,
               keyType='ENTREZID',
               pAdjustMethod = "BH",
               pvalueCutoff =0.01,
               qvalueCutoff = 0.01,
               ont="CC", #"MF", "CC",
               readable =T)
kkMF <- enrichGO(gene = gene$ENTREZID,
               OrgDb = org.Hs.eg.db,
               keyType='ENTREZID',
               pAdjustMethod = "BH",
               pvalueCutoff =0.01,
               qvalueCutoff = 0.01,
               ont="MF", #"MF", "CC",
               readable =T)
egoBP=simplify(all_BP_simp)
# egoBPf=data.frame(egoBP)

ego2=simplify(kk)
# ego3=data.frame(ego2)
#
ego22=simplify(kkMF)
# ego32=data.frame(ego22)

egoBP@result$Description=paste0(toupper(substring(egoBP@result$Description, 1, 1)),substring(egoBP@result$Description, 2))
ego2@result$Description=paste0(toupper(substring(ego2@result$Description, 1, 1)),substring(ego2@result$Description, 2))
ego22@result$Description=paste0(toupper(substring(ego22@result$Description, 1, 1)),substring(ego22@result$Description, 2))

pBP <- dotplot(egoBP, color = "pvalue",font.size=18,showCategory = 20 )+ theme(axis.text = element_text(family="Times New Roman",),
                                                         legend.title=element_text(size=12,  family="Times New Roman"),
                                                          legend.text = element_text(size=8,  family="Times New Roman"),
                                                         # axis.text.x = element_text(family="Times New Roman",face = 'bold'),
                                                         # axis.text.y = element_text(family="Times New Roman",face = 'bold'),
                                                         axis.title.x=element_text(family="Times New Roman",))#showCategory=30,
p1 <- dotplot(ego2, color = "pvalue",font.size=18)+ theme(axis.text = element_text(family="Times New Roman",face = 'bold'),
                                                         legend.title=element_text(size=12,  family="Times New Roman",face = 'bold'),
                                                          legend.text = element_text(size=8,  family="Times New Roman",face = 'bold'),
                                                         # axis.text.x = element_text(family="Times New Roman",face = 'bold'),
                                                         # axis.text.y = element_text(family="Times New Roman",face = 'bold'),
                                                         axis.title.x=element_text(family="Times New Roman",face = 'bold'))#showCategory=30,
# p2 <- dotplot(kegg, showCategory=20,color = "pvalue") + ggtitle("dotplot for KEGG")
p2 <- dotplot(ego22, color = "pvalue",font.size=18)+ theme(axis.text = element_text(family="Times New Roman",face = 'bold'),
                                                         legend.title=element_text(size=12,  family="Times New Roman",face = 'bold'),
                                                           legend.text = element_text(size=8,  family="Times New Roman",face = 'bold'),
                                                         # axis.text.x = element_text(family="Times New Roman",face = 'bold'),
                                                         # axis.text.y = element_text(family="Times New Roman",face = 'bold'),
                                                         axis.title.x=element_text(family="Times New Roman",face = 'bold'))#showCategory=30,
# # plot.grid(p1,p2,labels=("CC""MF"),align='h')
# p <- grid.arrange(p1, p2, ncol = 2)+theme(axis.text = element_text(family="Times New Roman",face = 'bold'),
#                                                          legend.title=element_text(size=12,  family="Times New Roman",face = 'bold'),
#                                                          axis.text.x = element_text(family="Times New Roman",face = 'bold'),
#                                                          axis.text.y = element_text(family="Times New Roman",face = 'bold'),
#                                                          axis.title.x=element_text(family="Times New Roman",face = 'bold'))
# ggsave('dotplot_BP_0.05_466_back_BP_5varibales_ols.jpg',pBP,dpi = 150, width = 250, height = 200, units = "mm")
ggsave('bubble_0.05_1731_back_BP_5varibales_ols.png',pBP, dpi = 300, width = 250, height = 250,  units = "mm")
ggsave('dotplot_CC_0.05_1731_5varibales_ols.png',p1,dpi = 300, width = 250, height = 200, units = "mm")
ggsave('dotplot_MF_0.05_1731_5varibales_ols.png', p2,dpi = 300, width = 250, height = 200, units = "mm")

