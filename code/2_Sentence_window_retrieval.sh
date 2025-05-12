python 2_Sentence_window_retrieval.py \
--model_type api \
--api_key 此处输入API KEY \
--embedding_model_path 本地嵌入模型路径，例如'../../Model/bge-large-en-v1.5' \
--rerank_model_path 本地嵌入模型路径，例如'../../Model/bge-reranker-base' \
--data_path '../data/Elon.txt'
