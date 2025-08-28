import os
import re
import warnings
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 添加LangSmith追踪导入
from langsmith import Client
from langsmith.run_helpers import traceable


# 过滤distutils相关的弃用警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*distutils.*")

# 定义图的状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 创建DuckDuckGo搜索工具
@tool
@traceable
def web_search(query: str) -> str:
    """使用DuckDuckGo搜索网络以获取相关信息"""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)


# 创建语言模型
# 使用您提供的本地模型配置
llm = ChatOpenAI(
    api_key=os.environ.get("LLM_API_KEY_QWQ", "local-qwen2.5-72b-little-brother"),
    base_url=os.environ.get("LLM_BASE_URL_QWQ", "http://10.8.50.33:8814/v1"),
    model="qwq-32b-preview",
    temperature=0
)


# 定义系统提示词，指导模型如何使用工具
SYSTEM_PROMPT = """
你是一个智能助手，可以使用工具来获取信息以回答用户问题。

你可以使用以下工具：
- web_search: 使用DuckDuckGo搜索网络以获取相关信息

当用户提出问题时，你应该：
1. 首先判断是否需要搜索网络来获取信息
2. 如果需要，直接调用web_search工具
3. 根据搜索结果回答用户问题

注意：只在确实需要外部信息时才使用搜索工具。
"""


# 定义图节点
@traceable
def chatbot(state: State):
    """聊天机器人节点，负责处理用户消息并决定是否使用工具"""
    # 在消息历史前添加系统提示
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    try:
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        error_message = f"抱歉，我在处理您的请求时遇到了问题: {str(e)}"
        return {"messages": [SystemMessage(content=error_message)]}


@traceable
def should_continue(state: State) -> str:
    """判断是否需要继续调用工具"""
    messages = state["messages"]
    if not messages:
        return END
        
    last_message = messages[-1]
    # 检查最后一条消息的内容是否表示需要搜索
    content = getattr(last_message, 'content', '').lower()
    
    # 启发式方法：检查是否包含搜索意图的关键词
    search_indicators = [
        "让我搜索一下",
        "我来查查",
        "搜索",
        "查找",
        "查询"
    ]
    
    for indicator in search_indicators:
        if indicator in content:
            return "tools"
            
    # 检查是否明确提到了使用搜索工具
    if "web_search" in content or "网络搜索" in content:
        return "tools"
            
    return END


# 创建工具节点
tool_node = ToolNode([web_search])

# 创建图
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# 添加边
graph_builder.add_edge(START, "chatbot")
# 使用自定义的条件函数
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")

# 编译图
graph = graph_builder.compile()


# 添加LangSmith追踪配置
def run_chatbot():
    """运行聊天机器人"""
    print("基于DuckDuckGo的网络搜索问答系统")
    print("输入 'quit' 或 'exit' 退出程序")
    print("-" * 40)
    
    # 检查LangSmith配置
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not langchain_api_key:
        print("⚠️  注意: 未配置LANGCHAIN_API_KEY环境变量，将不会进行LangSmith追踪")
        print("   如需启用追踪，请设置环境变量:")
        print("   export LANGCHAIN_API_KEY=your-api-key")
        print("   export LANGCHAIN_TRACING_V2=true")
        print("   export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com")
        print("-" * 40)
    
    # 初始化LangSmith客户端
    client = Client()
    
    while True:
        user_input = input("\n请输入您的问题: ").strip()
        if user_input.lower() in ["quit", "exit", ""]:
            print("再见！")
            break
            
        # 处理用户输入
        messages = [HumanMessage(content=user_input)]
        try:
            # 添加追踪配置
            config = RunnableConfig(
                callbacks=[],
                tags=["duckduckgo-search-agent"],
                metadata={
                    "user_input": user_input,
                    "session_id": "default_session"
                }
            )
            
            # 使用追踪运行图
            for chunk in graph.stream({"messages": messages}, config=config):
                for key, value in chunk.items():
                    if key == "chatbot":
                        # 打印AI的回复
                        response_content = value["messages"][-1].content
                        if response_content:
                            print(f"AI助手: {response_content}")
                    elif key == "tools":
                        print("正在搜索...")
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("请检查您的模型配置或网络连接")


if __name__ == "__main__":
    # 设置LangSmith环境变量（如果尚未设置）
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # 请确保设置您的LangSmith API密钥
    os.environ["LANGCHAIN_API_KEY"] = "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"
    
    run_chatbot()