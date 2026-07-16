from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory

from langchain_core.runnables import Runnable
from langchain_core.prompt_values import PromptValue

from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_ollama import OllamaEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_ollama import ChatOllama

def print_rendered_prompt(prompt_value) -> PromptValue:
    """打印渲染完成的提示词（仅接收PromptValue对象）"""
    print("=" * 20)
    # PromptValue 支持 to_string()
    print(prompt_value.to_string())
    print("=" * 20)
    return prompt_value


# 新增打印AI消息工具函数
def print_ai_message(msg):

    print("===== AI原始输出 =====")

    if isinstance(msg, tuple):
        msg = msg[0]

    print(msg.content)

    return msg

class RagService(object):
    def __init__(self):
        self.vector_service=VectorStoreService(
            embedding=OllamaEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知的参考资料为主,"
                 "简洁和专业的回答用户问题。参考资料：{context}。\n\n"),
                ("system","并且我提供用户对话的历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")

            ]
        )

        self.chat_model=ChatOllama(model=config.chat_model_name)

        self.chain=self.__get_chain()

    def __get_chain(self):
        """获取执行链条"""
        retriever=self.vector_service.get_retriever()


        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str=""
            for doc in docs:
                formatted_str +=f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value:dict) -> str:

            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | retriever | format_document
                }
                | RunnableLambda(format_for_prompt_template)
                | self.prompt_template
                | RunnableLambda(print_rendered_prompt) # 此时是PromptValue，可安全调用to_string()
                | self.chat_model
                | RunnableLambda(print_ai_message)
                | StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

if __name__ == '__main__':
    #session_id配置
    session_config={
        "configurable":{
            "session_id":"user_001",
        }
    }

    res=RagService().chain.invoke({"input":"东亚大树"},session_config)
    print(res)