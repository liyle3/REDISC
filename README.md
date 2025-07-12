# REDISC
ReDiSC: A Reparameterized Masked Diffusion Model for Scalable Node Classification with Structured Predictions



* Conda Environment

  > conda env create -f redisc.yml



* data

  > /REDISC/data/

  * five homophilic graph datasets: Cora, Citeseer, PubMed, Photo, Computer
  * two large graph in OGB: ogbn-arxiv and ogbn-products
  * three heterophilic graph datasets: Roman-empire, Amazon-ratings, Minesweeper



* how to run

  > python main.py --config-name CORA seed=0 diffusion.method=LP

  
