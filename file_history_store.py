import os,json
from typing import Sequence

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage, messages_to_dict
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory



#message_to_dict() 单个消息对象（BaseMessage类实例）->字典


def get_history(session_id):
    return FileChatMessageHistory(session_id,"./history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id=session_id
        self.storage_path=storage_path

        self.file_path=os.path.join(self.storage_path,self.session_id)

        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self, messages):

        all_messages = self.messages
        all_messages.extend(messages)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(
                messages_to_dict(all_messages),
                f,
                ensure_ascii=False,
                indent=2
            )


    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return messages_from_dict(data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)


model=ChatOllama(model="qwen3:4b")

str_parser=StrOutputParser()

def print_prompt(full_prompt):
    print("="*20 , full_prompt.to_string(),"="*20)
    return full_prompt


prompt=PromptTemplate.from_template(
    "你需要根据会话历史回应用户信息。对话历史：{chat_history}，用户提问：{input}，请回答。"
)

base_chain=prompt | print_prompt | model |  str_parser

store={}


conversation_chain=RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == '__main__':
    session_config={
        "configurable":{
            "session_id": "user_001"
        }
    }

    # res= conversation_chain.invoke({"input":"小明有两个猫。"},session_config)
    # print("第一次执行",res)
    #
    # res = conversation_chain.invoke({"input": "小刚有一只狗。"}, session_config)
    # print("第二次执行", res)
    #
    # res = conversation_chain.invoke({"input": "小李有三条蛇"}, session_config)
    # print("第三次执行",res)
    #
    # res = conversation_chain.invoke({"input": "总共有几个宠物"}, session_config)
    # print("第四次执行", res)

