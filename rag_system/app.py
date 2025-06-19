# 在导入任何其他库之前应用补丁
import os
import sys

# 设置环境变量，禁用Streamlit的文件监视功能
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 尝试修复PyTorch和Streamlit的兼容性问题
def patch_torch_for_streamlit():
    try:
        import torch
        
        # 创建一个安全的包装器，防止Streamlit访问torch.classes的特定属性
        class SafeClassesWrapper:
            def __init__(self, original_module):
                self._original = original_module
                
            def __getattr__(self, name):
                # 特殊处理可能导致问题的属性
                if name == '__path__' or name.startswith('_'):
                    if hasattr(self._original, name):
                        attr = getattr(self._original, name)
                        # 如果是一个对象并且有_path属性，返回一个安全的代理
                        if hasattr(attr, '_path'):
                            return type('SafePath', (), {'_path': []})
                        return attr
                    return None
                return getattr(self._original, name)
        
        # 打补丁到torch.classes
        if hasattr(torch, 'classes'):
            torch.classes = SafeClassesWrapper(torch.classes)
            print("已应用PyTorch与Streamlit兼容性补丁")
    except Exception as e:
        print(f"应用PyTorch补丁时出错: {e}")

# 应用补丁
patch_torch_for_streamlit()


import streamlit as st
import json
import traceback
import sys
import os

# 在导入任何PyTorch相关模块前先应用补丁
if 'fix_torch_streamlit' not in sys.modules:
    try:
        from fix_torch_streamlit import patch_torch_for_streamlit
        patch_torch_for_streamlit()
    except ImportError:
        st.warning("未找到PyTorch-Streamlit兼容性补丁模块，程序可能会不稳定")

# 禁用文件监视以防止段错误
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# 现在导入其他模块
from src.document_loader import DocumentLoader
from src.retriever import Retriever
from src.reranker import Reranker
from src.llm import LLM
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_RERANK_TOP_K
from src.services.retrieval_service import RetrievalService
from src.services.reranking_service import RerankingService
from src.services.generation_service import GenerationService

# 页面配置
st.set_page_config(
    page_title="机器学习知识问答系统",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """初始化会话状态"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'llm' not in st.session_state:
        st.session_state.llm = LLM()
    if 'documents' not in st.session_state:
        st.session_state.documents = []

    if 'retrieval_service' not in st.session_state:
        st.session_state.retrieval_service = None
    if 'reranking_service' not in st.session_state:
        st.session_state.reranking_service = None
    if 'generation_service' not in st.session_state:
        st.session_state.generation_service = None

    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'reranker' not in st.session_state:
        st.session_state.reranker = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'references' not in st.session_state:
        st.session_state.references = []

def safe_import_torch_modules():
    """安全导入PyTorch相关模块，防止崩溃"""
    try:
        # 只有在需要时才导入这些模块
        import torch
        import sentence_transformers
        import faiss
        
        # 检查CUDA可用性但不打印过多信息
        cuda_available = torch.cuda.is_available()
        device = 'cuda' if cuda_available else 'cpu'
        
        st.session_state.models_loaded = True
        return True
    except Exception as e:
        st.error(f"加载机器学习库时出错: {str(e)}")
        st.error(traceback.format_exc())
        return False

# 更新 app.py 中的 chat_interface 函数
def chat_interface():
    """优化后的多轮对话界面，修复HTML渲染问题"""
    import streamlit.components.v1 as components
    
    st.header("🤖 机器学习助手")
    st.markdown("我是专业的机器学习助手，可以回答机器学习、深度学习、数据科学相关的问题。")
    
    # 初始化会话状态
    if 'references_history' not in st.session_state:
        st.session_state.references_history = []
    
    # 显示聊天历史
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果是助手消息并且有对应的参考资料，显示参考资料（默认折叠）
            if message["role"] == "assistant" and idx//2 < len(st.session_state.references_history):
                refs = st.session_state.references_history[idx//2]
                if refs:
                    
                    # 使用深色主题的参考资料展示区
                    with st.expander("📚 查看参考资料", expanded=False):
                        # 使用components.html确保HTML正确渲染
                        references_html = f"""
                        <div style="background-color: #212121; padding: 15px; border-radius: 10px;">
                            <h2 style="color: #FFFFFF; margin-top: 0; margin-bottom: 20px; font-size: 24px;">参考文献</h2>
                        </div>
                        """
                        
                        for ref_idx, article in enumerate(refs, 1):
                            title = article.get('title', '未知标题')
                            link = article.get('link', '#')
                            preview = article.get('content_preview', '无摘要')
                            read_count = article.get('read_count', '无阅读数据')
                            like_count = article.get('like_count', '无点赞数据')
                            
                            references_html += f"""
                            <div style="background-color: #2D2D2D; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #2196F3;">
                                <div style="margin-bottom: 10px;">
                                    <h3 style="color: #2196F3; margin: 0; font-size: 18px;">{ref_idx}. <a href="{link}" target="_blank" style="color: #2196F3; text-decoration: none; font-weight: bold;">{title}</a></h3>
                                </div>
                                
                                <div style="color: #BDBDBD; margin-bottom: 12px; font-size: 14px;">
                                    {preview[:200]}{'...' if len(preview) > 200 else ''}
                                </div>
                                
                                <div style="display: flex; gap: 10px;">
                                    <div style="background-color: rgba(33, 150, 243, 0.2); color: #64B5F6; padding: 4px 10px; border-radius: 4px; font-size: 12px;">
                                        📖 {read_count}
                                    </div>
                                    <div style="background-color: rgba(76, 175, 80, 0.2); color: #81C784; padding: 4px 10px; border-radius: 4px; font-size: 12px;">
                                        👍 {like_count}
                                    </div>
                                </div>
                            </div>
                            """
                        
                        # 使用components.html正确渲染HTML
                        components.html(references_html, height=300, scrolling=True)
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成助手回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考并查询相关资料..."):
                response = st.session_state.llm.chat_response(st.session_state.messages)
                
                # 显示回答内容
                st.markdown(response["content"])
                
                # 保存消息到历史记录
                st.session_state.messages.append({"role": "assistant", "content": response["content"]})
                
                # 保存当前回答的参考资料
                st.session_state.references_history.append(response["references"])
                
                # 如果有参考资料，显示在回答下方
                if response["references"]:
                                        
                    # 使用深色主题的参考资料展示区
                    with st.expander("📚 查看参考资料", expanded=False):
                        # 使用components.html确保HTML正确渲染
                        references_html = f"""
                        <div style="background-color: #212121; padding: 15px; border-radius: 10px;">
                            <h2 style="color: #FFFFFF; margin-top: 0; margin-bottom: 20px; font-size: 24px;">参考文献</h2>
                        </div>
                        """
                        
                        for ref_idx, article in enumerate(response["references"], 1):
                            title = article.get('title', '未知标题')
                            link = article.get('link', '#')
                            preview = article.get('content_preview', '无摘要')
                            read_count = article.get('read_count', '无阅读数据')
                            like_count = article.get('like_count', '无点赞数据')
                            
                            references_html += f"""
                            <div style="background-color: #2D2D2D; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #2196F3;">
                                <div style="margin-bottom: 10px;">
                                    <h3 style="color: #2196F3; margin: 0; font-size: 18px;">{ref_idx}. <a href="{link}" target="_blank" style="color: #2196F3; text-decoration: none; font-weight: bold;">{title}</a></h3>
                                </div>
                                
                                <div style="color: #BDBDBD; margin-bottom: 12px; font-size: 14px;">
                                    {preview[:200]}{'...' if len(preview) > 200 else ''}
                                </div>
                                
                                <div style="display: flex; gap: 10px;">
                                    <div style="background-color: rgba(33, 150, 243, 0.2); color: #64B5F6; padding: 4px 10px; border-radius: 4px; font-size: 12px;">
                                        📖 {read_count}
                                    </div>
                                    <div style="background-color: rgba(76, 175, 80, 0.2); color: #81C784; padding: 4px 10px; border-radius: 4px; font-size: 12px;">
                                        👍 {like_count}
                                    </div>
                                </div>
                            </div>
                            """
                        
                        # 使用components.html正确渲染HTML
                        components.html(references_html, height=300, scrolling=True)
    
    # 清除对话按钮
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("清除对话历史"):
            st.session_state.messages = []
            st.session_state.references_history = []
            st.rerun()

def pdf_qa_interface():
    """PDF文档问答界面 - 使用微服务架构"""
    st.header("📄 PDF文档问答")
    
    # 显示服务状态
    show_service_status()
    
    # 参数配置
    with st.expander("⚙️ 参数配置", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.slider("文本块大小", 200, 1000, DEFAULT_CHUNK_SIZE, 50)
            chunk_overlap = st.slider("文本块重叠", 0, 200, DEFAULT_CHUNK_OVERLAP, 10)
        with col2:
            retrieval_top_k = st.slider("检索文档数", 10, 30, 20, 5)
            alpha = st.slider("向量检索权重", 0.0, 1.0, 0.7, 0.1)
        with col3:
            coarse_top_k = st.slider("粗排保留数", 5, 15, 10, 1)
            fine_top_k = st.slider("精排保留数", 1, 8, 3, 1)
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "上传PDF文档",
        type=['pdf'],
        accept_multiple_files=True,
        help="支持上传多个PDF文件"
    )
    
    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文件")
        
        # 处理文档并初始化服务
        if st.button("处理文档并初始化服务", key="process_docs"):
            process_documents_and_init_services(
                uploaded_files, chunk_size, chunk_overlap, 
                retrieval_top_k, coarse_top_k, fine_top_k
            )
    
    # 问答界面
    if all_services_ready():
        st.markdown("---")
        st.markdown("### 📚 微服务化文档问答")
        
        query = st.text_input("请输入您的问题：")
        
        if query:
            perform_microservice_qa(query, alpha)

def show_service_status():
    """显示服务状态"""
    st.markdown("### 🔧 服务状态")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.retrieval_service and st.session_state.retrieval_service.is_initialized:
            st.success("🔍 检索服务：运行中")
            metrics = st.session_state.retrieval_service.get_metrics()
            st.metric("检索调用次数", metrics['total_calls'])
            if metrics['total_calls'] > 0:
                st.metric("平均耗时", f"{metrics['avg_time']:.3f}s")
        else:
            st.error("🔍 检索服务：未初始化")
    
    with col2:
        if st.session_state.reranking_service and st.session_state.reranking_service.is_initialized:
            st.success("📊 重排序服务：运行中")
            metrics = st.session_state.reranking_service.get_metrics()
            st.metric("重排序调用次数", metrics['total_calls'])
            if metrics['total_calls'] > 0:
                st.metric("平均耗时", f"{metrics['avg_time']:.3f}s")
        else:
            st.error("📊 重排序服务：未初始化")
    
    with col3:
        if st.session_state.generation_service and st.session_state.generation_service.is_initialized:
            st.success("💬 生成服务：运行中")
            metrics = st.session_state.generation_service.get_metrics()
            st.metric("生成调用次数", metrics['total_calls'])
            if metrics['total_calls'] > 0:
                st.metric("平均耗时", f"{metrics['avg_time']:.3f}s")
        else:
            st.error("💬 生成服务：未初始化")

def process_documents_and_init_services(uploaded_files, chunk_size, chunk_overlap, 
                                      retrieval_top_k, coarse_top_k, fine_top_k):
    """处理文档并初始化所有服务"""
    # 确保ML库已加载
    if not st.session_state.models_loaded and not safe_import_torch_modules():
        st.error("机器学习模块初始化失败，请刷新页面重试")
        return
    
    with st.spinner("处理文档中..."):
        try:
            # 加载文档
            loader = DocumentLoader(chunk_size, chunk_overlap)
            documents = loader.load_pdf_documents(uploaded_files)
            
            if not documents:
                st.error("未能从上传的文档中提取到有效文本")
                return
            
            st.session_state.documents = documents
            st.success(f"文档处理完成，共生成 {len(documents)} 个文本块")
            
            # 初始化服务
            init_success = True
            
            # 初始化检索服务
            st.session_state.retrieval_service = RetrievalService(documents, retrieval_top_k)
            result = st.session_state.retrieval_service.call("test")  # 测试初始化
            if not result['success']:
                init_success = False
# 初始化重排序服务
            st.session_state.reranking_service = RerankingService(coarse_top_k, fine_top_k)
            result = st.session_state.reranking_service.call("test", [])  # 测试初始化
            if not result['success']:
                init_success = False
            
            # 初始化生成服务
            st.session_state.generation_service = GenerationService()
            result = st.session_state.generation_service.call("test", [])  # 测试初始化
            if not result['success']:
                init_success = False
            
            if init_success:
                st.success("🎉 所有微服务初始化完成！")
            else:
                st.warning("⚠️ 部分服务初始化失败，请检查配置")
                
        except Exception as e:
            st.error(f"文档处理或服务初始化失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def all_services_ready() -> bool:
    """检查所有服务是否就绪"""
    return (
        st.session_state.retrieval_service and st.session_state.retrieval_service.is_initialized and
        st.session_state.reranking_service and st.session_state.reranking_service.is_initialized and
        st.session_state.generation_service and st.session_state.generation_service.is_initialized
    )

def perform_microservice_qa(query: str, alpha: float):
    """执行微服务化的问答流程"""
    with st.spinner("正在处理您的问题..."):
        # 创建结果展示容器
        results_container = st.container()
        
        with results_container:
            # 显示处理流程
            st.markdown("#### 🔄 处理流程")
            
            # 步骤1：检索
            step1_placeholder = st.empty()
            step1_placeholder.info("步骤1: 混合检索中...")
            
            retrieval_result = st.session_state.retrieval_service.call(query, alpha)
            
            if not retrieval_result['success']:
                step1_placeholder.error(f"检索失败: {retrieval_result['error']}")
                return
            
            retrieved_docs = retrieval_result['data']
            retrieval_time = retrieval_result['metrics']['execution_time']
            
            step1_placeholder.success(f"✅ 步骤1完成: 检索到 {len(retrieved_docs)} 个文档 (耗时: {retrieval_time:.3f}s)")
            
            # 步骤2：重排序
            step2_placeholder = st.empty()
            step2_placeholder.info("步骤2: 两阶段重排序中...")
            
            reranking_result = st.session_state.reranking_service.call(query, retrieved_docs)
            
            if not reranking_result['success']:
                step2_placeholder.error(f"重排序失败: {reranking_result['error']}")
                return
            
            rerank_data = reranking_result['data']
            reranking_time = reranking_result['metrics']['execution_time']
            
            step2_placeholder.success(
                f"✅ 步骤2完成: 粗排→{rerank_data['stage_info']['coarse_count']}个, "
                f"精排→{rerank_data['stage_info']['fine_count']}个 (耗时: {reranking_time:.3f}s)"
            )
            
            # 步骤3：生成
            step3_placeholder = st.empty()
            step3_placeholder.info("步骤3: 生成回答中...")
            
            final_docs = rerank_data['fine_ranked']
            generation_result = st.session_state.generation_service.call(query, final_docs)
            
            if not generation_result['success']:
                step3_placeholder.error(f"生成失败: {generation_result['error']}")
                return
            
            answer = generation_result['data']
            generation_time = generation_result['metrics']['execution_time']
            
            step3_placeholder.success(f"✅ 步骤3完成: 回答生成完毕 (耗时: {generation_time:.3f}s)")
            
            # 显示结果
            st.markdown("---")
            st.markdown("#### 📝 生成的回答")
            st.write(answer)
            
            # 显示详细的检索和重排序结果
            show_detailed_results(query, retrieved_docs, rerank_data, final_docs)
            
            # 显示性能统计
            show_performance_summary(retrieval_time, reranking_time, generation_time)

def show_detailed_results(query: str, retrieved_docs, rerank_data, final_docs):
    """显示详细的检索和重排序结果"""
    with st.expander("🔍 详细处理结果", expanded=False):
        tab1, tab2, tab3 = st.tabs(["原始检索结果", "粗排结果", "精排结果"])
        
        with tab1:
            st.markdown(f"**混合检索结果** (共{len(retrieved_docs)}个)")
            for idx, doc in enumerate(retrieved_docs[:10], 1):  # 只显示前10个
                granularity = doc.metadata.get('granularity', 'paragraph')
                source = doc.metadata.get('filename', doc.metadata.get('title', '未知来源'))
                st.markdown(f"**{idx}. [{granularity}级别] {source}**")
                st.markdown(f"```\n{doc.page_content[:200]}...\n```")
        
        with tab2:
            coarse_docs = rerank_data['coarse_ranked']
            st.markdown(f"**粗排结果** (从{len(retrieved_docs)}个筛选到{len(coarse_docs)}个)")
            for idx, doc in enumerate(coarse_docs, 1):
                granularity = doc.metadata.get('granularity', 'paragraph')
                source = doc.metadata.get('filename', doc.metadata.get('title', '未知来源'))
                st.markdown(f"**{idx}. [{granularity}级别] {source}**")
                st.markdown(f"```\n{doc.page_content[:200]}...\n```")
        
        with tab3:
            st.markdown(f"**精排结果** (最终{len(final_docs)}个)")
            for idx, doc in enumerate(final_docs, 1):
                granularity = doc.metadata.get('granularity', 'paragraph')
                source = doc.metadata.get('filename', doc.metadata.get('title', '未知来源'))
                st.markdown(f"**{idx}. [{granularity}级别] {source}**")
                st.markdown(f"```\n{doc.page_content[:300]}...\n```")

def show_performance_summary(retrieval_time: float, reranking_time: float, generation_time: float):
    """显示性能摘要"""
    with st.expander("📊 性能统计", expanded=False):
        total_time = retrieval_time + reranking_time + generation_time
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("检索耗时", f"{retrieval_time:.3f}s")
        with col2:
            st.metric("重排序耗时", f"{reranking_time:.3f}s")
        with col3:
            st.metric("生成耗时", f"{generation_time:.3f}s")
        with col4:
            st.metric("总耗时", f"{total_time:.3f}s")
        
        # 性能分析图表
        st.markdown("**耗时分布**")
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame({
            '阶段': ['检索', '重排序', '生成'],
            '耗时(秒)': [retrieval_time, reranking_time, generation_time],
            '占比(%)': [
                retrieval_time/total_time*100,
                reranking_time/total_time*100,
                generation_time/total_time*100
            ]
        })
        
        fig = px.pie(df, values='耗时(秒)', names='阶段', title='各阶段耗时分布')
        st.plotly_chart(fig, use_container_width=True)

def knowledge_base_interface():
    """知识库展示界面"""
    st.header("📚 机器学习知识库")
    
    # 加载文章数据
    loader = DocumentLoader()
    articles = loader.get_knowledge_base_articles()
    
    if not articles:
        st.error("无法加载知识库文章")
        return
    
    # 搜索和筛选
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 搜索文章", placeholder="输入关键词搜索...")
    with col2:
        sort_by = st.selectbox("排序方式", ["按标题", "按阅读量", "按点赞数"])
    
    # 筛选文章
    filtered_articles = articles
    if search_term:
        filtered_articles = [
            article for article in articles
            if search_term.lower() in article['title'].lower() or 
               search_term.lower() in article.get('content_preview', '').lower()
        ]
    
    # 排序
    if sort_by == "按阅读量":
        def get_read_count(x):
            count = x.get('read_count', '0').replace('阅读 ', '')
            if count.endswith('k'):
                return int(float(count[:-1]) * 1000)
            elif count.endswith('w'):
                return int(float(count[:-1]) * 10000)
            else:
                return int(count) if count.isdigit() else 0
        
        filtered_articles.sort(key=get_read_count, reverse=True)
    elif sort_by == "按点赞数":
        filtered_articles.sort(key=lambda x: int(x.get('like_count', '0').replace('赞', '')), reverse=True)
    else:
        filtered_articles.sort(key=lambda x: x['title'])
    
    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总文章数", len(articles))
    with col2:
        st.metric("搜索结果", len(filtered_articles))
    with col3:
        st.metric("可用文档", len([a for a in articles if a.get('article_id')]))
    
    # 分页显示
    articles_per_page = 10
    total_pages = (len(filtered_articles) - 1) // articles_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("选择页面", range(1, total_pages + 1))
    else:
        page = 1
    
    start_idx = (page - 1) * articles_per_page
    end_idx = start_idx + articles_per_page
    page_articles = filtered_articles[start_idx:end_idx]
    
    # 显示文章
    for article in page_articles:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### [{article['title']}]({article['link']})")
                if article.get('content_preview'):
                    st.markdown(f"**摘要**: {article['content_preview'][:200]}...")
            
            with col2:
                if article.get('read_count'):
                    st.markdown(f"📖 {article['read_count']}")
                if article.get('like_count'):
                    st.markdown(f"👍 {article['like_count']}")
                if article.get('collect_count'):
                    st.markdown(f"⭐ {article['collect_count']}")
            
            st.divider()

def main():
    # 初始化
    initialize_session_state()
    
    # 侧边栏导航
    with st.sidebar:
        st.title("🤖 ML问答系统")
        page = st.selectbox(
            "选择功能",
            ["多轮对话", "PDF文档问答", "知识库浏览"]
        )
        
        st.markdown("---")
        st.markdown("""
        ### 系统说明
        - **多轮对话**: 与机器学习助手进行对话
        - **PDF文档问答**: 上传PDF并进行问答
        - **知识库浏览**: 浏览机器学习文章库
        """)
        
    
    # 主界面
    if page == "多轮对话":
        chat_interface()
    elif page == "PDF文档问答":
        pdf_qa_interface()
    elif page == "知识库浏览":
        knowledge_base_interface()

if __name__ == "__main__":
    try:
        # 设置一些环境变量，减少PyTorch的日志输出
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 运行应用
        main()
    except Exception as e:
        st.error(f"应用程序发生错误: {str(e)}")
        st.error(traceback.format_exc())