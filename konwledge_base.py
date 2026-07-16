"""
知识库
"""

import ssl

# 保存原函数，但这里我们不调用它，直接新建一个不验证的上下文
_original_create_default_context = ssl.create_default_context

def patched_create_default_context(*args, **kwargs):
    # 创建一个不加载系统证书的上下文
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# 全局替换
ssl.create_default_context = patched_create_default_context

import streamlit as st
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime



def check_md5(md5_str: str):
    md5_str = md5_str.strip()  # 统一去除首尾空白，避免文件换行符差异
    if not os.path.exists(config.md5_path):
        return False

    with open(config.md5_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stripped_lines = [line.strip() for line in lines]
        for line in stripped_lines:
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    # 写入前检查，避免重复
    if check_md5(md5_str):
        return
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    # 只去除文件末尾的换行符（\n 和 \r），保留其他空白
    if input_str.endswith('\n'):
        input_str = input_str[:-1]
    if input_str.endswith('\r'):
        input_str = input_str[:-1]
    # 或者更彻底，去除末尾所有空白，但保留开头的空白
    # input_str = input_str.rstrip()
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，若是存在则自动跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma=Chroma(
            collection_name=config.collection_name, #数据库表名
            embedding_function = OllamaEmbeddings(model="nomic-embed-text") ,
            persist_directory=config.persist_directory #数据库本地存储文件夹
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  #分割后文本最大长度
            chunk_overlap=config.chunk_overlap, #连续文本段之间的字符重叠数量
            separators=config.separators, #自然段落划分的符号
            length_function=len, #使用python自带的len函数长度统计
        ) #文本分割器的对象

        # 启动同步
        self.sync_md5()

    def sync_md5(self):

        """
        防止：
        chroma存在数据
        md5为空

        """

        data = self.chroma.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        md5_set = set()

        for meta in data["metadatas"]:

            if meta and "md5" in meta:
                md5_set.add(meta["md5"])

        with open(
                config.md5_path,
                "w",
                encoding="utf-8"
        ) as f:

            for md5 in md5_set:
                f.write(
                    md5 + "\n"
                )


    def upload_by_str(self , data , filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        #先得到传入字符串的md5值
        md5_hex=get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小曹"
        }

        # 修复参数名：metdatas → metadatas
        self.chroma.add_texts( #内容加载到向量库中了
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]  # 关键修复点
        )

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"


if __name__ == '__main__':
    service=KnowledgeBaseService()


