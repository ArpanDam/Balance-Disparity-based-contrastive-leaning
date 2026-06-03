# Balance-Disparity-based-contrastive-leaning
This code describes the Graph contrastive learning for citation network


Download the files from this folder : https://www.dropbox.com/scl/fo/j8ga1klbs0t091nc8u5x5/AHGv7OLkulcSECXgPvUcbZw?rlkey=ze26jve7zrqnmisdvi007ypg6&st=do83o3ho&dl=0


Run PNA.py to get the embeddings of the nodes. These embeddings are stored in h1.These embeddings are used to find the top k influential authors


# Finding top k influential authors
Run the code inside the folder Seed_finder as : python seed_nodes.py 10 0.4 to find top 10 influential authors

Here, 10 is the number of top k inclusive influencers, and 0.4 is the threshold beta.
