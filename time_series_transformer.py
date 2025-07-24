'''
总结：Transformer 在网络安全领域的优势

优势	说明
强大序列建模能力	适合建模攻击链条、用户行为、网络流等时序数据
多模态兼容	能处理日志、流量、文件、文本等多种数据类型
易于预训练 & 迁移学习	可微调 BERT 或 ViT 等安全相关任务
更强的上下文理解	识别复杂行为模式（如 APT、社会工程攻击）

模型总结：常用于网络安全的 Transformer 类型

名称	用途	来源 / 模型类型
LogBERT	日志异常检测	结构化日志 + BERT
AnomalyTransformer	时间序列异常检测	改进 attention + series decomposition
TransIDS	入侵检测	网络流量分类
Malformer	恶意软件检测	二进制序列建模
EMBERT	可执行文件语义表示	二进制预训练模型
CyberBERT	安全语义分析（CVE/IOT）	领域知识增强版 BERT

总结：Transformer 在安防领域的高频用途

安防方向	技术核心	是否已有部署
静态图像识别	ViT / DETR / Swin Transformer	✅ 大量部署
视频行为识别	TimeSformer / ViViT	⏳ 正在快速部署
声音分析	AST / Whisper / Wav2Vec	✅ 有用例（如枪声检测）
多模态融合	CLIP / GPT-4V / BLIP-2	⏳ 发展中（AI 驱动智能安防）

开源框架推荐

框架 / 项目	用途
MMSelfSup	支持 ViT 视觉预训练
MMDetection	DETR/YOLO 安防图像检测
TimeSformer	视频 Transformer
BLIP-2	图文联合分析

具体模型推荐（按任务）

安防任务	推荐模型（现成 or 微调）
图像中检测武器	✅ ViT + Faster R-CNN、YOLOv8 + ViT、DETR
视频中识别攻击行为	✅ TimeSformer、VideoMAE、ViViT
枪声检测	✅ AST（Audio Spectrogram Transformer）
图文联合理解报警信息	✅ CLIP、BLIP-2、GPT-4V

总结：Transformer 的最多用途分类

分类领域	用途关键词	模型代表
自然语言处理	文本生成、理解、问答、翻译	GPT, BERT, T5, BART
计算机视觉	图像分类、检测、生成	ViT, Swin, DETR, DALL·E
多模态	图文匹配、视频理解、音频分析	CLIP, GPT-4V, Gemini, BLIP
编程领域	自动写代码、代码理解	Codex, CodeLlama
时间序列	预测、异常检测、信号建模	Informer, Autoformer
推荐系统	用户序列建模	BERT4Rec, SASRec

实际中可用的时间序列 Transformer 模型

模型	特点	用途
Informer	用稀疏注意力建模长序列，高效预测	预测、检测
Autoformer	自适应分解时间序列，识别趋势/季节性	复杂时间序列预测
Anomaly Transformer	显性建模正常与异常之间的“关联强度”	强化异常检测
TS-T5	用于时间序列“语言化”理解 + 多任务预测	预测 + 解释性任务

Input Sequence [x1, x2, ..., xT]
     ↓
+-------------------------+
| Embedding + Positional |
+-------------------------+
     ↓
Transformer Encoder Stack (Self-Attention)
     ↓
     ↓
Detection Head / Forecasting Head
'''
class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        return self.proj(x)  # [batch, seq_len, d_model]
def sinusoidal_position_encoding(seq_len, d_model):
    import math
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe  # shape: [seq_len, d_model]

'''
encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
'''

class ClassifierHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):  # x shape: [batch, seq_len, d_model]
        cls_token = x[:, 0, :]  # 假设用第一个 token 表示全序列
        return self.fc(cls_token)  # [batch, 2]

import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 1)  # 二分类：异常/正常

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        x = self.embed(x)  # [batch, seq_len, d_model]
        #x = embed(x) + pos_embed  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to [batch, seq_len, d_model]
        logits = self.head(x)  # [batch, seq_len, 1]
        return logits.squeeze(-1)  # 每个时间点一个异常概率

#代码示例：简易 Transformer 日志异常检测
'''
推荐数据集

名称	描述	链接
HDFS	Hadoop 日志，有标签	https://github.com/logpai/loghub
BGL	BlueGene/L 系统日志	同上
Thunderbird	NASA 系统日志	同上
LANL	登录/认证事件（攻击/正常）	https://csr.lanl.gov/data/cyber1/

一句话总结：

✅ 小型 Transformer（LogBERT 等）适合结构化日志建模和高效部署；
✅ LLM 更适合多样化、非结构化、复杂语义的日志分析和语义推理。

场景适配建议

场景	推荐模型	理由
✅ 高吞吐日志系统（如大数据平台）	LogBERT / AnomalyTransformer	高效，结构化，部署方便
✅ 工业控制系统日志（少样本）	Self-Attention + One-Class	训练数据稀缺，结构清晰
✅ 原始、非结构日志	GPT-4 / Claude / LLama2-Chat	能直接理解自然语言日志
✅ 高级安全分析（威胁溯源）	GPT + Prompt / RAG	可回答“攻击发生在哪”“异常原因是什么”
✅ 自动 SOC 工单生成	GPT + 日志总结 Prompt	可生成处理建议、事件摘要

总结

推荐	说明
LogBERT / AnomalyTransformer	🚀 高效、成熟、结构化，适合工业部署
GPT-4 / Claude / LLaMA	🧠 智能语义分析，适合高级异常解释、灵活分析场景
'''

import torch
import torch.nn as nn

class LogTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))  # max_len=512

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)  # output: anomaly score (regression or sigmoid)

    def forward(self, x):  # x: [batch, seq_len]
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)  # [batch, seq_len, d_model]
        x_cls = x[:, -1, :]  # use last token's output
        return torch.sigmoid(self.fc(x_cls))
'''
总结

推荐	说明
LogBERT / AnomalyTransformer	🚀 高效、成熟、结构化，适合工业部署
GPT-4 / Claude / LLaMA	🧠 智能语义分析，适合高级异常解释、灵活分析场景

请求序列转化为 Embedding 输入
离散特征（如请求路径、UA、方法） ➜ 用 embedding
数值特征（请求间隔、payload 大小）➜ 标准化输入
时间特征（如小时、周几） ➜ sin/cos 或 one-hot 编码

五、自监督训练技巧（无标签也能训练）

Masked Log Modeling（MLM）：像 BERT 一样随机 mask 日志 token，预测它
Next Log Prediction：输入前 N 条日志，预测第 N+1 条的模板 ID
Sequence Validity Classification：判断一个序列是否是“正常日志顺序”

 推荐开源方案

项目/库	功能
BotD by Fingerprint	浏览器指纹识别，检测自动化工具
OpenWAF	支持基于规则+行为检测的 Web 防火墙
DeepLog	日志序列 Transformer 异常检测
ELK Stack	日志采集、展示、配合模型使用

总结

目标	推荐方案
高频 Bot 检测	Transformer 模型 + 请求序列建模
多模态 Bot 检测	Mouse + JS + Log 多模态 Transformer
可解释性增强	LLM + Prompt 生成异常解释
原始 UA/Headers 检测	微调 BERT/GPT 模型，理解 Header 特征语义
'''
class BotRequestTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls = nn.Linear(d_model, 1)  # 二分类输出：是否为 Bot

    def forward(self, x):  # x shape: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.encoder(x)
        out = x[:, -1]  # 最后一个请求表示整体行为
        return torch.sigmoid(self.cls(out))

'''
将离散特征（categorical features）转为 embedding 是现代机器学习特别是在 Transformer、DNN 中的核心步骤之一
'''
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=4, embedding_dim=8)  # 4 类 UA → 8 维 embedding
input = torch.tensor([1])  # UA 是 "Safari"
embedded = embedding(input)  # 输出形状: [1, 8]

'''
多个离散特征联合建模
'''
class CategoricalEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ua_embed = nn.Embedding(100, 16)
        self.geo_embed = nn.Embedding(50, 8)
        self.method_embed = nn.Embedding(5, 4)

    def forward(self, ua_id, geo_id, method_id):
        ua_vec = self.ua_embed(ua_id)
        geo_vec = self.geo_embed(geo_id)
        method_vec = self.method_embed(method_id)
        return torch.cat([ua_vec, geo_vec, method_vec], dim=-1)  # [batch_size, total_dim]

# 每个时间步的 embedding 维度 = 连续特征 + 离散 embedding 总维度
step_input = torch.cat([
    numerical_features,       # [batch, seq_len, num_numerical_features]
    categorical_embed_output  # [batch, seq_len, embedding_dim]
], dim=-1)

# 投影到 Transformer 的 d_model 维度
x = self.linear_input(step_input)

'''
nn.Embedding 是 PyTorch 中用于**将离散类别映射为连续向量（embedding 向量）**的一个神经网络层。
是的，在默认情况下，nn.Embedding 里的参数会随着模型的训练一起被优化。这就是它的强大之处 —— 它不仅是一个查表操作，而且是一个**“可学习”的查表层**
'''

import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=4, embedding_dim=8)

input = torch.tensor([2])  # 表示 "Safari"
output = embedding(input)

print(output.shape)  # 输出: torch.Size([1, 8])

