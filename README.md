# WTRC: Spatially-Weighted Temporal Rich Club

**Identifying rich clubs in spatiotemporal interaction networks**
 
![WTRC](https://github.com/GeoDS/WTRC/blob/master/wi_wtrc_ttrc_horizontal_ave.png)
![WTRC Example](https://github.com/GeoDS/WTRC/blob/master/flow_snapshots_nolabels.png)

**Abstract:** 
Spatial networks are widely used in various fields to represent and analyze interactions or relationships between locations or spatially distributed entities or objects. While existing studies have proposed methods for hub identification and community detection in spatial networks, relatively few have focused on quantifying the strength or density of connections shared within a community of hubs across space and time. Borrowing from network science, there is a relevant concept known as the 'rich club' phenomenon, which describes the tendency of 'rich' nodes to form densely interconnected sub-networks. Although there are established methods to quantify topological, weighted, and temporal rich clubs individually, there is limited research on measuring the rich club effect in spatially-weighted temporal networks, which could be particularly useful for studying dynamic spatial interaction networks. To address this gap, we introduce the spatially-weighted temporal rich club (WTRC), a metric that quantifies the strength and consistency of connections between rich nodes in a spatiotemporal network. Additionally, we present a unified rich club framework that distinguishes the WTRC effect from other rich club effects, providing a way to measure topological, weighted, and temporal rich club effects together. Through two case studies of human mobility networks at different spatial scales, we demonstrate how the WTRC is able to identify significant weighted temporal rich club effects, whereas the unweighted equivalent in the same network either fails to detect a rich club effect or inaccurately estimates its significance. In each case study, we explore the spatial layout and temporal variations revealed by the WTRC analysis, showcasing its particular value in studying spatiotemporal interaction networks. This research offers new insights into the study of spatiotemporal networks, with critical implications for applications such as transportation, redistricting, and epidemiology.

## Paper

If you find our code on WTRC useful for your research, you may cite our paper:

Kruse, J., Gao, S.*, Ji, Y., Levin, K., Huang, Q., and Mayer, K. (2025).  [Identifying rich clubs in spatiotemporal interaction networks](https://arxiv.org/abs/2210.08041). Annals of the American Association of Geographers. X, X, 1-20.


```
@article{kruse2025identifying,
  title={Identifying rich clubs in spatiotemporal interaction networks},
  author={Kruse, Jacob and Gao, Song and Ji, Yuhan and Levin, Keith and Huang, Qunying and Mayer, Kenneth},
  journal={Annals of the American Association of Geographers},
  volume={x},
  number={x},
  pages={x},
  year={2025},
  publisher={Taylor and Francis}
}
```

You may also be interested in the original TRC reserch: 

Pedreschi, N., Battaglia, D., & Barrat, A. (2022). [The temporal rich club phenomenon](https://www.nature.com/articles/s41567-022-01634-8). *Nature Physics*, 18(8), 931-938.
Github: [https://github.com/nicolaPedre/Temporal-Rich-Club](https://github.com/nicolaPedre/Temporal-Rich-Club)

```
@article{pedreschi2022temporal,
  title={The temporal rich club phenomenon},
  author={Pedreschi, Nicola and Battaglia, Demian and Barrat, Alain},
  journal={Nature Physics},
  volume={18},
  number={8},
  pages={931--938},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

## Requirements
WTRC uses the following packages with Python 3.12.0:
numpy==1.26.0
pandas==2.1.1
shapely==2.0.1
geopandas=0.14.0
matplotlib-base=3.8.0

A full list of the packages and package versions can be found in the trc_env.yml file.

![image](https://github.com/user-attachments/assets/131a9e04-3795-45b4-819a-64b7aae7b799)

## Usage
There are two demo files: WTRC_example.ipynb, and TTRC_example.ipynb. To distinguish the weighted temporal rich club effects from the topological temporal rich club effects, you can run both and compare them. While the files are mostly similar, the WTRC and the TTRC use different randomization methods to prepare the null graphs, and all edge weights are set to 1 in the TTRC.
