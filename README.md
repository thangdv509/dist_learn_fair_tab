1. Train model trên original data
   → models/dataset/best_model.pth

2. Cluster original data để hiểu distribution
   python cluster_and_sample.py \
       --model-path models/german-credit-data/best_model_20260128_144037.pth\
       --dataset german-credit-data \
       --output-dir sampled_data/german-credit-data
   → sampled_data/sampled_sentences.csv (các câu mẫu từ clusters)

3. Sinh synthetic data từ sampled sentences
   (dùng sampled_sentences.csv để guide generation)

4. Embed synthetic data để tính fairness
   python embed_data.py \
       --model-path models/dataset/best_model.pth \
       --input-data synthetic_data.csv \
       --output-dir embeddings/synthetic
   → embeddings/synthetic/synthetic_embeddings_z_c.npy
   → embeddings/synthetic/synthetic_embeddings_z_d.npy

5. Tính fairness metrics trên z_c và z_d của synthetic data