# Retrieval Augmented Generation (RAG) Intro Project 🤖🔍📝

**RAG教学示例项目**  
1. 项目使用了[LlamaIndex](https://github.com/run-llama/llama_index)框架演示了RAG的基本流程，也欢迎使用其它框架如[LangChain](https://www.langchain.com/)等进行实验，或者自己依照RAG的流程独立编写相关流程。
2. 项目示例支持使用本地模型和外部API调用两种方法。项目示例中部分使用了[阿里云](https://help.aliyun.com/zh/dashscope/developer-reference/?spm=a2c4g.11186623.0.0.644e9b6em7thMV)提供的模型接口(新用户拥有1,000,000tokens免费额度，需自行申请API Key)，也可尝试使用其它免费的API，需在代码中相应位置做替换。
3. 项目示例中使用qwen2.5-14b-instruct（API调用）为生成模型，同时使用本地的bge-large-en-v1.5为embedding模型，bge-reranker-base为reranker模型，可根据实际情况进行调整。

## Project Structure 📂

#### 1. `README.md`: 项目总览

#### 2. **code**:  文件夹下包含三个教学实验，三个实验均有各自python文件以及可直接运行的shell脚本文件，示例文件中使用样例数据。此外，文件夹内还有基于自动合并索引方法处理CRAG数据的示例，以及对于RAG效果的评测脚本，以下为简要说明：
- `1_Basic_RAG_Pipeline`: RAG基础的pipeline 
  ![image](data/llamaindex_rag_overview.png)
- `2_Sentence_window_retrieval`: 将文档拆分成小的组块（chunk），将输入的问题（query）和检索结果的上下文（chunks）都拼接到prompt中，输入给生成模型得到最终的回答
  <img src="data/llamaindex_SentenceWindowRetrieval_overview.png" width="400" height="300"><img src="data/llamaindex_SentenceWindowRetrieval_example.png" width="400" height="300">

- `3_Auto-merging_Retrieval`: 将文档按照块大小拆分成不同层级的节点，在检索时使用叶子节点检索，然后检查父节点包含的子节点中被检索到的比例，高于一定阈值时将父节点作为检索结果，否则将被检索到的子节点作为检索结果，AutoMergingRetrieval的详细介绍可参考 https://zhaozhiming.github.io/2024/03/19/auto-merging-rag/
![image](data/llamaindex_AutoMergingRetrieval_example.png)

- `crag.sh`: 基于 自动合并索引 方法对CRAG数据进行处理，得到query、answer以及模型的输出pred并保存，便于后续评测
- `metric.py`: 评测脚本，计算模型生成内容与标准答案的BLEU和Rouge-l指标
- `model_response.py`: 提供了API访问和本地部署LLM两种方式，选择本地部署的同学可以进一步改造代码，使用`vllm`框架等加速推理
#### 3. **data**: 实验所需的语料，包括
  - `Elon.txt`: 示例文件 `Elon.txt`
  - CRAG数据：提供了200条训练子集（crag_data_200.jsonl）和去除html标签后的2735条全量数据集(BGE.jsonl)。每条数据包含**query**、**可能和query相关的五个网页内容**、**query对应的答案**。如有训练需求可自行将CRAG全量训练集切分作为你自己的训练集和验证集。此外，200条训练子集、全量未去除html标签和去除html标签后的2735条数据三种版本下载地址为https://www.modelscope.cn/datasets/zhaofangkun/CRAG_cleaned/summary
  - 可以选择使用BGE.jsonl数据集用于BGE微调.

## Getting Started 🚀

1. 克隆或下载项目仓库到本地：
```shell
  git clone https://github.com/ZhaoFangkun1/NLP_RAG_Demo.git
```
2. 准备环境
```shell
  conda create -n rag python=3.10
  conda activate rag
  pip install peft
  pip install requests
  pip install llama_index
  pip install llama-index-embeddings-huggingface
```
3. 进入`code`文件夹，在各实验代码中调整好模型来源（API KEY或模型本地地址），依次运行三种实验脚本
```shell
cd code
1. sh 1_Basic_RAG_Pipeline.sh
2. sh 2_Sentence_window_retrieval.sh
3. sh 3_Auto-merging_Retrieval.sh
```
4. 更改数据为提供的CRAG数据集进行测试，同时也可尝试其它的数据集。
```shell
sh crag.sh
python metric.py
```
5. 鼓励进行模型训练或流程上的改进
6. 参考配置(不要求完全一致，可在出现 依赖库版本 相关问题时参考)
```
python==3.10
peft==0.15.2
requests==2.32.3
llama_index==0.12.35
llama-index-embeddings-huggingface==0.5.4
```

## 项目要求：下面三项内容，至少选择其二开展进一步探索并完成实验报告
1. 检索策略优化：尝试更多的组块拆分策略、检索算法和重排算法
2. 微调BGE的嵌入模型或（和）重排模型：
  - 自行将CRAG全量训练集进行划分，微调BGE嵌入以及重排模型，可参考 https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
  - 微调时需要构造query对应的chunk正负样例，但是CRAG数据集并未给出query对应的chunk正例（只给出了与每个query可能相关的网页、每个query对应的答案），因此需要设计方案为query构造正例（除人工标注外，还可通过prompt LLM进行query和chunk的相关性判定）
3. 对LLM进行进一步预训练或微调：
  - 根据自己的算力情况，使用给定的CRAG数据对LLM做进一步的预训练或有监督微调
  - 为了使用方便，我们同时更新了`model_response.py`，如要使用lora权重进行推理，只需要在`MyLocalLLM`类的初始化方法中使用`LocalPeftModel`类即可。
   - 对使用的LLM框架不做限制，可参考[llama-factory](https://github.com/hiyouga/LLaMA-Factory)、[megatron-lm](https://github.com/NVIDIA/Megatron-LM)框架，以及阿里进行二次封装之后的[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)框架等
