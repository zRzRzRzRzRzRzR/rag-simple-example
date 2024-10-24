# Simple RAG System

## 部署并运行模型

### 1. 使用docker构建数据库

创建 PostgreSQL 数据库。

```shell
sudo docker run --name rag -e POSTGRES_PASSWORD=dianjiao29 -p 54320:5432 -d postgres
```

接着，安装向量化插件

```shell

安装向量数据库插件
```shell
sudo docker exec -it rag bash
sudo apt update 
sudo apt install postgresql-17-pgvector
```

接着，执行`init_tabel`中的SQL命令，完成表的初始化。

### 2. Python环境配置

请确保你的服务器拥有英伟达GPU，并正确适配了最新的 CUDA 和 cuDNN 以及绑定的 torch版本。按照要求创建虚拟环境并安装依赖。

```shell
pip install -r requirements.txt
```

前往 `app.py` 并设置对应的API密钥和模型名称。这里默认配置为：

+ LLM: 智谱AI的 `GLM-4-AirX` 模型，其最大优点是相应快。
+ Embedding: `embedding-3` 作为向量模型。
+ PDF转换服务：`http://localhost:5001/convert_pdf`, 请确保PDF转换服务已经启动。详见第三步。
+ 默认API密钥：请填写对应的API密钥，这里默认使用智谱AI的模型，因此需要填写智谱AI的key，如果你使用别的大模型，请更换对应的key以及url。
+ OSS配置：请填写对应的OSS配置，这里默认使用阿里云OSS，因此需要填写对应的key，bucket_name以及endpoint。

```
default_api_key = 'ZhipuAI Keys'
default_api_url = 'https://open.bigmodel.cn/api/paas/v4'
default_llm_model = 'glm-4-airx'
default_embedding_model = 'embedding-3'
flask_pdf_converter_url = "http://localhost:5001/convert_pdf"
os.environ['OSS_ACCESS_KEY_ID'] = 'oss access key'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'oss access secret'
bucket_name = 'zrzrzr'
endpoint = 'oss-cn-beijing.aliyuncs.com'
```

### 3. 启动PDF2MarkDown服务

本项目直接使用[marker-pdf](https://github.com/VikParuchuri/marker) 作为PDF转换工具，因此需要先启动PDF转换服务。

```shell
python pdf_convert.py
```

你可以运行 `test\test_pdf_marker.py` 验证是否正常启动服务。

```shell
python test\test_pdf_marker.py
```

如果正常，你将会

### 5. 运行程序

```shell
python app.py
```

你将会看到一个由 gradio部署的前端UI，能够可视化的选择模型，选择文件，以及和大模型进行RAG对话。

+ 对 Textbook.pdf 可能需要数分钟才能完成分词。

## 局限性

本项目仅为一个简单的RAG系统，因此存在一些局限性：

1. 无并发
2. 无法进行批量推理
3. 初始化未配置config等配置文件，直接在python代码中修改。
4. 没有对图片进行处理。

## 开源协议

本仓库遵循 Apache-2.0 开源协议。