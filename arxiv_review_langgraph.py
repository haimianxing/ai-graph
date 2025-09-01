"""
基于arxiv的论文深度搜索和综述生成系统（LangGraph版本）
该系统使用LangGraph的状态图实现工作流，并可视化显示状态图
"""

import os
import re
import arxiv
import requests
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
try:
    import PyPDF2
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    print("警告: PyPDF2未安装，将使用论文摘要而不是完整内容")
from typing import List, Dict, Annotated, TypedDict, Optional
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
# 添加LangSmith追踪导入
from langsmith import traceable, Client
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 添加LangGraph相关导入
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
import warnings

# 添加图像显示相关导入
from IPython.display import Image, display

max_papers = 20

# 过滤distutils相关的弃用警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*distutils.*")


# 定义状态模型
class GraphState(TypedDict):
    """LangGraph状态模型"""
    topic: str  # 用户输入的研究主题
    query: str  # 翻译和扩展后的查询词
    papers: List[Dict]  # 搜索到的论文列表
    analysis: str  # 论文分析结果
    review: str  # 生成的综述报告
    filename: str  # 保存的文件名
    messages: Annotated[list, add_messages]  # 消息历史


class ArxivReviewGenerator:
    def __init__(self):
        """初始化arxiv综述生成器"""
        # 创建语言模型
        self.llm = ChatOpenAI(
            api_key=os.environ.get("LLM_API_KEY_QWQ", "local-qwen2.5-72b-little-brother"),
            base_url=os.environ.get("LLM_BASE_URL_QWQ", "http://10.8.50.14:8814/v1"),
            model="qwq-32b-preview",
            temperature=0
        )

        # 初始化MapReduce链
        self.map_reduce_chain = self._create_map_reduce_chain()
        
        # 创建工作流图
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_map_reduce_chain(self):
        """创建MapReduce链用于处理论文摘要"""
        # Map阶段提示词
        map_template = """以下是论文摘要内容:
{context}
请根据这些摘要内容，提取每篇论文的核心要点，包括：
1. 研究主题和目标
2. 使用的方法或技术 （要求：详细具体，有细节，要深刻具体）
3. 主要发现或贡献 （要求：详细具体，有细节，要深刻具体）
4. 研究局限性（如果有） （要求：详细具体，有细节，要深刻具体）
请提供创新处的专业具体详细的总结。

摘要总结:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = map_prompt | self.llm | StrOutputParser()

        # Reduce阶段提示词
        reduce_template = """以下是多篇论文的摘要总结:
{context}
请根据这些摘要总结，进行综合分析，要求：
1. 按主题或方法对论文进行分类  （要求：详细具体，有细节，要深刻具体）
2. 识别研究中的共同点和差异 （要求：详细具体，有细节，要深刻具体）
3. 指出共同研究趋势 
4. 分析当前研究的空白点或挑战

综合分析报告:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()

        # 返回map和reduce链
        return {"map_chain": map_chain, "reduce_chain": reduce_chain}

    def _translate_query_node(self, state: GraphState) -> GraphState:
        """将中文查询翻译成英文，以便更好地在arXiv中搜索"""
        query = state["topic"]
        # 如果查询已经是英文，直接返回
        if re.match(r'^[a-zA-Z\s]+$', query):
            print("📝 输入查询词已经是英文，无需翻译")
            return {"query": query}

        prompt = f"""
请将以下中文翻译成适合在学术搜索引擎中使用的英文术语：
中文：{query}

要求：
1. 翻译成准确的学术英文术语
2. 如果有多个可能的翻译，提供最常见的那个
3. 只返回翻译结果，不要包含其他内容
"""

        try:
            print(f"🤖 正在调用LLM进行查询词翻译: '{query}'")
            start_time = time.time()
            translation = self.llm.invoke(prompt)
            end_time = time.time()
            translated_query = translation.content.strip()
            print(f"✅ 查询词翻译完成，耗时: {end_time - start_time:.2f}秒")
            print(f"🔍 查询词翻译: '{query}' -> '{translated_query}'")
            return {"query": translated_query}
        except Exception as e:
            print(f"⚠️ 翻译查询词时出错: {str(e)}, 使用原始查询词")
            return {"query": query}

    def _expand_query_node(self, state: GraphState) -> GraphState:
        """扩展查询词，添加相关术语以提高搜索效果"""
        query = state["query"]
        prompt = f"""
请为以下学术主题提供相关的英文关键词，用于在arXiv中搜索：
主题：{query}

要求：
1. 提供 5个最相关的英文关键词或短语
2. 包括该领域的标准术语和可能的同义词
3. 用OR连接这些关键词，形成一个搜索表达式
4. 只返回搜索表达式，不要包含其他解释

例如：machine learning OR deep learning OR neural networks OR AI
"""

        try:
            print(f"🤖 正在调用LLM进行查询词扩展: '{query}'")
            start_time = time.time()
            expansion = self.llm.invoke(prompt)
            end_time = time.time()
            expanded_query = expansion.content.strip()
            print(f"✅ 查询词扩展完成，耗时: {end_time - start_time:.2f}秒")
            print(f"🔍 查询词扩展: '{query}' -> '{expanded_query}'")
            return {"query": expanded_query}
        except Exception as e:
            print(f"⚠️ 扩展查询词时出错: {str(e)}, 使用原始查询词")
            return {"query": query}

    def _search_papers_node(self, state: GraphState) -> GraphState:
        """搜索arxiv论文"""
        query = state["query"]
        print(f"🔍 开始搜索论文，查询词: '{query}'")
        
        try:
            print(f"🔍 正在使用查询词搜索论文: '{query}'")
            client = arxiv.Client()

            # 计算两年前的日期
            two_years_ago = datetime.now().replace(tzinfo=None) - timedelta(days=365*2)

            search = arxiv.Search(
                query=query,
                max_results=max_papers * 2,  # 搜索更多论文以补偿筛选
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )

            # 更安全地获取结果，处理可能的空页面问题
            results = []
            try:
                results = list(client.results(search))
            except Exception as e:
                if "unexpectedly empty" in str(e):
                    print("⚠️ 搜索结果页面为空，尝试使用更简单的查询")
                    # 使用更简单的查询重试
                    simple_query = state["topic"]  # 回退到原始主题作为查询词
                    print(f"🔄 使用简化查询重试: '{simple_query}'")
                    simple_search = arxiv.Search(
                        query=simple_query,
                        max_results=max_papers,
                        sort_by=arxiv.SortCriterion.Relevance,
                        sort_order=arxiv.SortOrder.Descending
                    )
                    results = list(client.results(simple_search))
                else:
                    raise e

            papers = []

            # 筛选最近两年的论文
            filtered_results = []
            for result in results:
                published_date = result.published.replace(tzinfo=None)
                if published_date >= two_years_ago:
                    filtered_results.append(result)

            print(f"📅 筛选最近两年论文: {len(results)} -> {len(filtered_results)}")

            # 使用线程池并行处理论文
            with ThreadPoolExecutor(max_workers=25) as executor:
                # 提交所有任务
                future_to_index = {
                    executor.submit(self._process_paper_result, result): i
                    for i, result in enumerate(filtered_results[:max_papers])
                }

                # 收集结果
                for future in tqdm(as_completed(future_to_index),
                                 total=len(future_to_index),
                                 desc="处理论文",
                                 colour="GREEN"):
                    try:
                        paper = future.result(timeout=600)
                        papers.append(paper)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"⚠️ 处理第 {index + 1} 篇论文时出错: {e}")

            print(f"✅ 论文搜索完成，共找到 {len(papers)} 篇论文")

            # 使用规划器评估论文相关性并调整关键词
            papers = self._planner_filter_papers(papers, state["topic"], max_papers)

            return {"papers": papers}
        except Exception as e:
            print(f"搜索论文时出错: {str(e)}")
            # 返回空列表而不是中断整个流程
            return {"papers": []}

    def _process_paper_result(self, result):
        """处理单个论文结果的辅助方法"""
        print(f"📄 获取到论文: {result.title}")
        paper = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
            "categories": result.categories
        }

        # 尝试提取PDF内容
        if PDF_PROCESSING_AVAILABLE:
            try:
                print(f"📄 正在提取论文PDF内容: {result.title}")
                paper["full_content"] = self._extract_pdf_content(result.pdf_url)
                print(f"✅ 论文PDF内容提取完成: {result.title}")
            except Exception as e:
                print(f"⚠️ 提取PDF内容时出错: {str(e)}, 使用摘要内容")
                paper["full_content"] = result.summary
        else:
            paper["full_content"] = result.summary
            print(f"📄 使用论文摘要内容: {result.title}")

        return paper

    def _planner_filter_papers(self, papers: List[Dict], original_query: str, max_results: int) -> List[Dict]:
        """使用规划器评估论文相关性并过滤不相关论文"""
        if not papers:
            return papers

        print(f"🧠 规划器开始评估 {len(papers)} 篇论文的相关性")

        # 评估每篇论文与原始查询的相关性
        relevant_papers = []
        irrelevant_papers = []

        for paper in papers:
            is_relevant = self._evaluate_paper_relevance(paper, original_query)
            if is_relevant:
                relevant_papers.append(paper)
            else:
                irrelevant_papers.append(paper)

        print(f"✅ 规划器评估完成: {len(relevant_papers)} 篇相关, {len(irrelevant_papers)} 篇不相关")

        # 如果相关论文数量不足，尝试调整关键词重新搜索
        if len(relevant_papers) < max_results * 0.7:  # 如果相关论文少于70%
            print(f"⚠️ 相关论文不足，正在调整关键词重新搜索...")
            additional_papers = self._search_additional_papers(original_query, max_results - len(relevant_papers), papers)
            relevant_papers.extend(additional_papers)

        # 返回相关论文，限制数量
        return relevant_papers[:max_results]

    def _evaluate_paper_relevance(self, paper: Dict, query: str) -> bool:
        """评估单篇论文与查询的相关性"""
        try:
            content = paper.get("full_content", paper["summary"])

            prompt = f"""
请评估以下论文与用户查询的相关性：

用户查询: {query}

论文标题: {paper['title']}
论文摘要: {paper['summary'][:1000]}...

请判断这篇论文是否与用户查询直接相关。考虑以下因素：
1. 论文主题是否与查询一致
2. 论文是否解决了查询中提到的问题或相关问题
3. 论文的方法或技术是否适用于查询领域

回答"相关"或"不相关"，并简要说明理由（不超过50字）。

例如：
相关 - 论文主题与查询完全匹配
不相关 - 论文属于完全不同领域
相关 - 虽然方法不同但解决相同问题
"""

            response = self.llm.invoke(prompt)
            result = response.content.strip().lower()

            # 判断是否相关
            is_relevant = "相关" in result or "relevant" in result

            if not is_relevant:
                print(f"🗑️  过滤不相关论文: {paper['title'][:50]}...")

            return is_relevant
        except Exception as e:
            print(f"⚠️ 评估论文相关性时出错: {str(e)}, 默认认为相关")
            return True  # 出错时默认认为相关

    def _search_additional_papers(self, original_query: str, needed_count: int, existing_papers: List[Dict]) -> List[Dict]:
        """根据现有论文分析结果，调整关键词并搜索更多相关论文"""
        try:
            # 从现有论文中提取关键词和主题
            existing_titles = [paper['title'] for paper in existing_papers]
            existing_summaries = [paper['summary'][:500] for paper in existing_papers]

            prompt = f"""
基于以下已检索到的论文，分析用户查询"{original_query}"的更精确关键词：

已检索到的论文标题:
{chr(10).join(existing_titles[:5])}

已检索到的论文摘要:
{chr(10).join(existing_summaries[:3])}

请分析这些论文的共同特征，为原始查询"{original_query}"提供3-5个更精确的英文关键词或短语，
以便检索到更相关的论文。只需要返回关键词，用OR连接，不需要解释。

例如：machine learning OR deep learning OR neural networks
"""

            response = self.llm.invoke(prompt)
            refined_query = response.content.strip()

            print(f"🔍 使用调整后的关键词搜索: {refined_query}")

            # 使用新关键词搜索论文
            client = arxiv.Client()
            two_years_ago = datetime.now().replace(tzinfo=None) - timedelta(days=365*2)

            search = arxiv.Search(
                query=refined_query,
                max_results=needed_count * 2,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )

            results = list(client.results(search))

            # 筛选最近两年的论文
            filtered_results = []
            for result in results:
                published_date = result.published.replace(tzinfo=None)
                if published_date >= two_years_ago:
                    filtered_results.append(result)

            print(f"📅 筛选最近两年论文: {len(results)} -> {len(filtered_results)}")

            # 处理新论文
            additional_papers = []
            for result in filtered_results[:needed_count]:
                try:
                    paper = self._process_paper_result(result)
                    # 再次检查相关性
                    if self._evaluate_paper_relevance(paper, original_query):
                        additional_papers.append(paper)
                except Exception as e:
                    print(f"⚠️ 处理补充论文时出错: {str(e)}")

            print(f"✅ 补充搜索完成，新增 {len(additional_papers)} 篇相关论文")
            return additional_papers[:needed_count]
        except Exception as e:
            print(f"⚠️ 补充搜索时出错: {str(e)}")
            return []

    @traceable
    def _extract_pdf_content(self, pdf_url: str) -> str:
        """从PDF链接提取内容"""
        try:
            print(f"📥 正在下载PDF文件: {pdf_url}")
            # 下载PDF文件
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            print("✅ PDF文件下载完成")

            # 使用PyPDF2读取PDF内容
            print("📄 正在解析PDF内容")
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # 提取前10页的内容（通常是摘要、引言和方法部分）
            content = ""
            pages_to_extract = min(20, len(pdf_reader.pages))
            print(f"📄 正在提取PDF前 {pages_to_extract} 页内容")

            # 使用线程池并行提取PDF页面内容
            page_contents = [""] * pages_to_extract
            with ThreadPoolExecutor(max_workers=25) as executor:
                # 提交所有页面提取任务
                future_to_page = {
                    executor.submit(self._extract_pdf_page, pdf_reader, i): i
                    for i in range(pages_to_extract)
                }

                # 收集结果
                for future in as_completed(future_to_page):
                    try:
                        page_index, page_text = future.result(timeout=120)
                        page_contents[page_index] = page_text
                    except Exception as e:
                        page_index = future_to_page[future]
                        print(f"⚠️ 提取第 {page_index} 页PDF内容时出错: {e}")

            content = "\n".join(page_contents)
            print("✅ PDF内容提取完成")
            return content
        except Exception as e:
            raise Exception(f"提取PDF内容失败: {str(e)}")

    def _extract_pdf_page(self, pdf_reader, page_index):
        """提取PDF单页内容的辅助方法"""
        page = pdf_reader.pages[page_index]
        return page_index, page.extract_text() + "\n"

    @traceable
    def _extract_core_content(self, full_content: str) -> str:
        """从论文完整内容中提取核心创新改进方法和实验数据部分"""
        try:
            # 使用LLM提取核心内容
            prompt = f"""
请从以下论文内容中提取核心创新改进方法和实验数据部分：

论文内容:
{full_content[:80000]}  # 限制长度以避免超出上下文窗口

请提取以下内容：
1. 论文的核心创新点或改进方法
2. 关键的实验数据和结果
3. 重要的性能指标和对比结果

要求：
- 保持原文的技术细节和数据准确性
- 只提取最关键和最有价值的信息 
- 保持简洁，总长度不超过80000字
- 以结构化的方式呈现提取的内容

提取结果:
"""

            print("🤖 正在调用LLM提取论文核心内容")
            start_time = time.time()
            extraction = self.llm.invoke(prompt)
            end_time = time.time()
            print(f"✅ 核心内容提取完成，耗时: {end_time - start_time:.2f}秒")
            return extraction.content.strip()
        except Exception as e:
            print(f"提取核心内容时出错: {str(e)}, 返回完整内容的前2000个字符")
            return full_content[:2000]

    def _analyze_papers_node(self, state: GraphState) -> GraphState:
        """分析论文并提取关键信息"""
        papers = state["papers"]
        if not papers:
            return {"analysis": "没有找到相关论文。"}

        print("📊 开始分析论文内容")
        # 使用MapReduce方法分析论文
        print("🔄 步骤1: 使用MapReduce方法分析论文")
        map_reduce_analysis = self.analyze_papers_with_map_reduce(papers)

        # 构建论文信息总结
        print("🔄 步骤2: 构建论文详细信息")
        paper_summaries = []

        # 使用线程池并行提取核心内容
        with ThreadPoolExecutor(max_workers=25) as executor:
            # 提交所有核心内容提取任务
            future_to_index = {
                executor.submit(self._extract_summary, paper, i): i
                for i, paper in enumerate(papers)
            }

            # 收集结果
            summary_results = []
            for future in tqdm(as_completed(future_to_index),
                             total=len(future_to_index),
                             desc="构建论文摘要",
                             colour="MAGENTA"):
                try:
                    index, summary = future.result(timeout=600)
                    summary_results.append((index, summary))
                except Exception as e:
                    index = future_to_index[future]
                    print(f"⚠️ 构建第 {index + 1} 篇论文摘要时出错: {e}")

            # 按索引排序以保持原始顺序
            summary_results.sort(key=lambda x: x[0])
            paper_summaries = [summary for _, summary in summary_results]

        prompt = f"""
请分析以下{len(papers)}篇论文并提取关键信息：

{chr(10).join(paper_summaries)}

请按以下要求进行分析：
1. 按研究主题或方法对论文进行分类  （要求：详细具体，有细节，要深刻具体）
2. 提取每篇论文的核心贡献  （要求：详细具体，有细节，要深刻具体）
3. 识别研究中的共同点和差异  （要求：详细具体，有细节，要深刻具体）
4. 指出研究趋势和发展方向 
"""

        try:
            print("🤖 正在调用LLM进行详细论文分析")
            start_time = time.time()
            analysis = self.llm.invoke(prompt)
            end_time = time.time()
            print(f"✅ 详细论文分析完成，耗时: {end_time - start_time:.2f}秒")
            # 结合MapReduce分析结果和LLM分析结果
            combined_analysis = f"""
MapReduce分析结果:
{map_reduce_analysis}

详细分析结果:
{analysis.content}
            """
            print("✅ 论文分析完成")
            return {"analysis": combined_analysis}
        except Exception as e:
            return {"analysis": f"分析论文时出错: {str(e)}"}

    def analyze_papers_with_map_reduce(self, papers: List[Dict]) -> str:
        """使用MapReduce方法分析论文摘要"""
        if not papers:
            return "没有找到相关论文。"

        print(f"🔄 开始使用MapReduce方法分析 {len(papers)} 篇论文")
        # 将论文摘要转换为Document对象

        # 使用线程池并行提取核心内容
        docs = []
        with ThreadPoolExecutor(max_workers=25) as executor:
            # 提交所有核心内容提取任务
            future_to_index = {
                executor.submit(self._create_document, paper, i, len(papers)): i
                for i, paper in enumerate(papers)
            }

            # 收集结果
            doc_results = []
            for future in tqdm(as_completed(future_to_index),
                             total=len(future_to_index),
                             desc="提取论文核心内容",
                             colour="BLUE"):
                try:
                    doc = future.result(timeout=600)
                    doc_results.append(doc)
                except Exception as e:
                    index = future_to_index[future]
                    print(f"⚠️ 提取第 {index + 1} 篇论文核心内容时出错: {e}")

            # 按索引排序以保持原始顺序
            doc_results.sort(key=lambda x: x[0])
            docs = [doc for _, doc in doc_results]

        try:
            # 使用MapReduce链处理文档
            map_chain = self.map_reduce_chain["map_chain"]
            reduce_chain = self.map_reduce_chain["reduce_chain"]

            # Map阶段：对每篇论文进行分析
            print("🧠 开始Map阶段：逐篇分析论文")
            map_results = []

            # 使用线程池并行分析论文
            with ThreadPoolExecutor(max_workers=25) as executor:
                # 提交所有分析任务
                future_to_index = {
                    executor.submit(self._analyze_single_paper, map_chain, doc, i): i
                    for i, doc in enumerate(docs)
                }

                # 收集结果
                analysis_results = []
                for future in tqdm(as_completed(future_to_index),
                                 total=len(future_to_index),
                                 desc="分析论文",
                                 colour="YELLOW"):
                    try:
                        index, result = future.result(timeout=600)
                        analysis_results.append((index, result))
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"⚠️ 分析第 {index + 1} 篇论文时出错: {e}")

                # 按索引排序以保持原始顺序
                analysis_results.sort(key=lambda x: x[0])
                map_results = [result for _, result in analysis_results]

            # Reduce阶段：综合分析所有结果
            print("🧠 开始Reduce阶段：综合分析所有论文")
            combined_context = "\n\n".join([f"论文分析 {i+1}:\n{result}" for i, result in enumerate(map_results)])
            print("🤖 正在调用LLM进行综合分析")
            start_time = time.time()
            final_result = reduce_chain.invoke({"context": combined_context})
            end_time = time.time()
            print(f"✅ 综合分析完成，耗时: {end_time - start_time:.2f}秒")

            print("✅ MapReduce分析完成")
            return final_result
        except Exception as e:
            return f"使用MapReduce分析论文时出错: {str(e)}"

    def _create_document(self, paper, index, total):
        """创建文档对象的辅助方法"""
        print(f"📄 处理论文 {index+1}/{total}: {paper['title']}")
        core_content = self._extract_core_content(paper.get("full_content", paper["summary"]))

        doc_content = f"""
论文 {index+1}:
标题: {paper['title']}
作者: {', '.join(paper['authors'][:3])} 等
发表日期: {paper['published']}
分类: {', '.join(paper['categories'])}
核心内容: {core_content}
        """
        return index, Document(page_content=doc_content)

    def _analyze_single_paper(self, map_chain, doc, index):
        """分析单篇论文的辅助方法"""
        print(f"🤖 正在调用LLM分析第 {index+1} 篇论文")
        start_time = time.time()
        result = map_chain.invoke({"context": doc.page_content})
        end_time = time.time()
        print(f"✅ 第 {index+1} 篇论文分析完成，耗时: {end_time - start_time:.2f}秒")
        return index, result

    def _extract_summary(self, paper, index):
        """提取论文摘要的辅助方法"""
        # 提取核心内容替代摘要
        core_content = self._extract_core_content(paper.get("full_content", paper["summary"]))

        summary = f"""
论文 {index+1}:
标题: {paper['title']}
作者: {', '.join(paper['authors'][:3])} 等
发表日期: {paper['published']}
分类: {', '.join(paper['categories'])}
核心内容: {core_content[:500]}...
        """
        return index, summary

    def _generate_review_node(self, state: GraphState) -> GraphState:
        """生成综述报告"""
        topic = state["topic"]
        papers = state["papers"]
        analysis = state["analysis"]
        
        if not analysis:
            return {"review": "没有分析内容可用于生成综述。"}

        print(f"📝 开始生成关于'{topic}'的综述报告")
        # 构建参考文献列表
        references = []
        for i, paper in enumerate(papers):
            ref = f"{i+1}. {', '.join(paper['authors'][:3])} et al. {paper['title']}. arXiv:{paper['entry_id'].split('/')[-1]}. {paper['published']}."
            references.append(ref)

        prompt = f"""
基于以下信息，现在请生成一篇完整的学术综述报告。

研究主题: {topic}

论文分析:
{analysis}

参考文献:
{chr(10).join(references)}

请按照以下结构生成综述报告，使用Markdown格式：

# 关于{topic}的文献综述

## 1. 引言
介绍研究领域和主题背景

## 2. 研究现状
专业按主题或方法分类讨论相关研究，并给出关注重点和拓展方向

## 3. 主要方法
专业总结研究方法和技术，给出专业犀利的点评和核心技术原理

## 4. 实用技术、框架结论
总结整体研究状况并提出实用技术、框架的专业观点

## 参考文献
{chr(10).join(references)}

要求：
- 内容专业、准确、连贯
- 结构清晰，逻辑性强
- 字数不少于5000字
"""

        try:
            print("🤖 正在调用LLM生成综述报告")
            start_time = time.time()
            review = self.llm.invoke(prompt)
            end_time = time.time()
            print(f"✅ 综述报告生成完成，耗时: {end_time - start_time:.2f}秒")
            return {"review": review.content}
        except Exception as e:
            return {"review": f"生成综述时出错: {str(e)}"}

    def _save_review_node(self, state: GraphState) -> GraphState:
        """保存综述报告到文件"""
        review_content = state["review"]
        topic = state["topic"]
        
        if not review_content:
            return {"filename": "没有内容可保存。"}

        # 清理文件名
        filename = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        if not filename:
            filename = "literature_review"

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.md"

        try:
            print(f"💾 正在保存综述报告到文件: {filename}")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(review_content)
            print(f"✅ 综述报告已保存到: {filename}")
            return {"filename": filename}
        except Exception as e:
            return {"filename": f"保存文件时出错: {str(e)}"}

    def _create_workflow(self):
        """创建工作流图"""
        # 定义图
        workflow = StateGraph(GraphState)

        # 添加节点
        workflow.add_node("translate_query", self._translate_query_node)
        workflow.add_node("expand_query", self._expand_query_node)
        workflow.add_node("search_papers", self._search_papers_node)
        workflow.add_node("analyze_papers", self._analyze_papers_node)
        workflow.add_node("generate_review", self._generate_review_node)
        workflow.add_node("save_review", self._save_review_node)

        # 添加边
        workflow.add_edge(START, "translate_query")
        workflow.add_edge("translate_query", "expand_query")
        workflow.add_edge("expand_query", "search_papers")
        workflow.add_edge("search_papers", "analyze_papers")
        workflow.add_edge("analyze_papers", "generate_review")
        workflow.add_edge("generate_review", "save_review")
        workflow.add_edge("save_review", END)

        return workflow

    def generate_review_for_topic(self, topic: str, max_papers: int = max_papers) -> str:
        """为指定主题生成综述报告"""
        print(f"🔍 正在搜索关于'{topic}'的论文...")
        
        # 创建初始状态
        initial_state = GraphState(
            topic=topic,
            query="",
            papers=[],
            analysis="",
            review="",
            filename="",
            messages=[]
        )
        
        # 执行工作流
        result = self.app.invoke(initial_state)
        
        print(f"✅ 找到 {len(result['papers'])} 篇相关论文")
        if not result['papers']:
            return "未能找到相关论文。"

        print("📊 论文分析完成")
        print("📝 综述报告生成完成")
        print("💾 报告已保存")
        
        return result["filename"]
        
    def visualize_workflow(self):
        """可视化工作流图"""
        try:
            # 生成图像
            image_data = self.app.get_graph().draw_mermaid_png()
            # 保存图像到文件
            with open("workflow_graph.png", "wb") as f:
                f.write(image_data)
            print("✅ 工作流图已保存到 workflow_graph.png")
            
            # 在Jupyter Notebook中显示图像
            try:
                display(Image("workflow_graph.png"))
            except:
                print("💡 提示：在Jupyter Notebook中可以显示图像")
                
            return "workflow_graph.png"
        except Exception as e:
            print(f"❌ 生成工作流图时出错: {str(e)}")
            return None


def run_chat_mode():
    """运行对话模式"""
    print("📚 基于arxiv的论文深度搜索和综述生成系统 (对话模式)")
    print("=" * 60)
    print("支持的命令:")
    print("  - 输入研究主题以生成文献综述")
    print("  - 'help' 查看帮助信息")
    print("  - 'quit' 或 'exit' 退出系统")
    print("=" * 60)

    # 初始化LangSmith客户端
    client = Client()

    # 创建综述生成器
    generator = ArxivReviewGenerator()
    
    # 显示工作流图
    print("🔄 正在生成工作流图...")
    graph_file = generator.visualize_workflow()
    if graph_file:
        print(f"📊 工作流图已生成: {graph_file}")

    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的研究主题或命令: ").strip()

            # 处理退出命令
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("👋 感谢使用，再见！")
                break

            # 处理帮助命令
            if user_input.lower() in ["help", "帮助"]:
                print("\n📖 帮助信息:")
                print("  1. 输入您感兴趣的研究主题（中文或英文）")
                print("  2. 系统将自动翻译并扩展查询词")
                print("  3. 系统会搜索相关论文并生成综述报告")
                print("  4. 报告将保存为Markdown文件")
                print("\n📝 示例主题:")
                print("   - 机器学习")
                print("   - 人工智能在医疗领域的应用")
                print("   - deep learning")
                continue

            # 处理空输入
            if not user_input:
                print("⚠️ 请输入有效的研究主题或命令")
                continue

            # 生成综述
            print(f"\n🚀 开始处理主题: {user_input}")
            filename = generator.generate_review_for_topic(user_input)
            print(f"\n🎉 文献综述已成功生成并保存至: {filename}")

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断信号，系统退出！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print("🔧 请检查您的输入或网络连接后重试")


def main():
    """主函数"""
    # 如果提供了命令行参数，则直接处理该主题
    import sys
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        generator = ArxivReviewGenerator()
        # 显示工作流图
        print("🔄 正在生成工作流图...")
        graph_file = generator.visualize_workflow()
        if graph_file:
            print(f"📊 工作流图已生成: {graph_file}")
            
        filename = generator.generate_review_for_topic(topic)
        print(f"🎉 文献综述已成功生成并保存至: {filename}")
        return

    run_chat_mode()


if __name__ == "__main__":
    # 设置LangSmith环境变量（如果尚未设置）
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # 请确保设置您的LangSmith API密钥
    os.environ["LANGCHAIN_API_KEY"] = "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"

    # 检查LangSmith配置
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not langchain_api_key:
        print("⚠️  注意: 未配置LANGCHAIN_API_KEY环境变量，将不会进行LangSmith追踪")
        print("   如需启用追踪，请设置环境变量:")
        print("   export LANGCHAIN_API_KEY=your-api-key")
        print("   export LANGCHAIN_TRACING_V2=true")
        print("   export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com")
        print("=" * 50)

    main()