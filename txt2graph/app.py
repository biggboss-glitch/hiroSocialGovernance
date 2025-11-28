"""
txt2graph 可视化界面
基于Streamlit和PyVis实现知识图谱可视化
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

from dotenv import load_dotenv
load_dotenv()

from text_extractor import extract_text, split_text_into_chunks
from graph_builder import ZepGraphBuilder, GraphData


# 页面配置
st.set_page_config(
    page_title="txt2graph - 知识图谱生成器",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono&display=swap');
    
    .main {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .stTitle {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .entity-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
        font-weight: 500;
    }
    
    .entity-Person { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; border: 1px solid #ff6b6b; }
    .entity-Company { background: rgba(78, 205, 196, 0.2); color: #4ecdc4; border: 1px solid #4ecdc4; }
    .entity-Organization { background: rgba(69, 183, 209, 0.2); color: #45b7d1; border: 1px solid #45b7d1; }
    .entity-Location { background: rgba(150, 206, 180, 0.2); color: #96ceb4; border: 1px solid #96ceb4; }
    .entity-Product { background: rgba(255, 238, 173, 0.2); color: #ffeead; border: 1px solid #ffeead; }
    .entity-Event { background: rgba(220, 198, 224, 0.2); color: #dcc6e0; border: 1px solid #dcc6e0; }
    .entity-Media { background: rgba(255, 183, 77, 0.2); color: #ffb74d; border: 1px solid #ffb74d; }
    
    .sidebar .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# 实体类型对应的颜色
ENTITY_COLORS = {
    "Person": "#ff6b6b",
    "Company": "#4ecdc4",
    "Organization": "#45b7d1",
    "Location": "#96ceb4",
    "Product": "#ffeead",
    "Event": "#dcc6e0",
    "Media": "#ffb74d",
}


def create_pyvis_graph(graph_data: GraphData) -> str:
    """
    创建PyVis图并返回HTML
    """
    # 创建网络图
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
    )
    
    # 配置物理引擎
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "Noto Sans SC, Arial"
            },
            "borderWidth": 2,
            "shadow": true
        },
        "edges": {
            "color": {
                "inherit": false,
                "color": "#555555",
                "highlight": "#667eea"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "continuous",
                "roundness": 0.2
            },
            "font": {
                "size": 10,
                "color": "#888888",
                "face": "Noto Sans SC, Arial"
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "stabilization": {
                "enabled": true,
                "iterations": 200
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # 构建节点UUID到名称的映射
    node_map = {node.uuid: node for node in graph_data.nodes}
    
    # 添加节点
    for node in graph_data.nodes:
        # 确定节点类型和颜色
        node_type = node.labels[0] if node.labels else "Unknown"
        color = ENTITY_COLORS.get(node_type, "#888888")
        
        # 构建工具提示
        title = f"<b>{node.name}</b><br>"
        title += f"<i>类型: {node_type}</i><br><br>"
        if node.summary:
            title += f"{node.summary[:200]}{'...' if len(node.summary) > 200 else ''}"
        
        # 根据节点类型调整大小
        size = 25 if node_type == "Person" else 30 if node_type in ["Company", "Organization"] else 20
        
        net.add_node(
            node.uuid,
            label=node.name,
            title=title,
            color=color,
            size=size,
            shape="dot",
        )
    
    # 添加边
    for edge in graph_data.edges:
        if edge.source_node_uuid in node_map and edge.target_node_uuid in node_map:
            # 构建边的工具提示
            title = edge.fact if edge.fact else edge.name
            
            net.add_edge(
                edge.source_node_uuid,
                edge.target_node_uuid,
                title=title,
                label=edge.name[:20] if edge.name else "",
            )
    
    # 生成HTML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
        os.unlink(f.name)
    
    return html_content


def display_stats(graph_data: GraphData):
    """显示图谱统计信息"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{len(graph_data.nodes)}</div>
            <div class="stats-label">实体节点</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{len(graph_data.edges)}</div>
            <div class="stats-label">关系边</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 统计实体类型分布
    type_counts = {}
    for node in graph_data.nodes:
        node_type = node.labels[0] if node.labels else "Unknown"
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{len(type_counts)}</div>
            <div class="stats-label">实体类型</div>
        </div>
        """, unsafe_allow_html=True)


def display_entity_list(graph_data: GraphData):
    """显示实体列表"""
    st.subheader("实体列表")
    
    # 按类型分组
    entities_by_type = {}
    for node in graph_data.nodes:
        node_type = node.labels[0] if node.labels else "Unknown"
        if node_type not in entities_by_type:
            entities_by_type[node_type] = []
        entities_by_type[node_type].append(node)
    
    # 创建标签页
    if entities_by_type:
        tabs = st.tabs(list(entities_by_type.keys()))
        
        for tab, (entity_type, entities) in zip(tabs, entities_by_type.items()):
            with tab:
                for entity in entities:
                    with st.expander(f"{entity.name}", expanded=False):
                        if entity.summary:
                            st.write(entity.summary)
                        if entity.attributes:
                            st.json(entity.attributes)


def main():
    # 标题
    st.title("txt2graph")
    st.markdown("*将文本转化为知识图谱*")
    
    # 侧边栏
    with st.sidebar:
        st.header("配置")
        
        # API Key
        api_key = st.text_input(
            "Zep API Key",
            type="password",
            value=os.environ.get("ZEP_API_KEY", ""),
            help="从 https://app.getzep.com 获取API Key"
        )
        
        if api_key:
            os.environ["ZEP_API_KEY"] = api_key
        
        st.divider()
        
        # 文件上传
        st.header("上传文件")
        uploaded_file = st.file_uploader(
            "支持 .txt, .md, .pdf 文件",
            type=["txt", "md", "pdf"],
            help="上传要转换为知识图谱的文本文件"
        )
        
        # 或者直接输入文本
        st.divider()
        st.header("或直接输入文本")
        text_input = st.text_area(
            "输入文本内容",
            height=150,
            placeholder="在此输入或粘贴文本..."
        )
        
        st.divider()
        
        # 高级设置
        with st.expander("高级设置"):
            chunk_size = st.slider(
                "文本分块大小",
                min_value=500,
                max_value=4000,
                value=2000,
                step=500,
                help="较小的块处理更稳定，较大的块包含更多上下文"
            )
            
            graph_name = st.text_input(
                "图谱名称",
                value="Knowledge Graph",
                help="为生成的图谱命名"
            )
        
        st.divider()
        
        # 生成按钮
        generate_btn = st.button("生成知识图谱", type="primary", use_container_width=True)
    
    # 主内容区
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = None
    
    if generate_btn:
        if not api_key:
            st.error("请先配置 Zep API Key")
            return
        
        # 获取文本内容
        text_content = None
        
        if uploaded_file:
            with st.spinner("正在提取文本..."):
                # 保存上传的文件到临时位置
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    text_content = extract_text(tmp_path)
                finally:
                    os.unlink(tmp_path)
        elif text_input:
            text_content = text_input
        else:
            st.warning("请上传文件或输入文本")
            return
        
        if text_content:
            st.info(f"提取了 {len(text_content)} 个字符的文本")
            
            # 进度显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 创建图谱构建器
                builder = ZepGraphBuilder(api_key=api_key)
                
                # 创建图谱
                status_text.text("创建图谱...")
                progress_bar.progress(10)
                graph_id = builder.create_graph(name=graph_name)
                
                # 设置本体
                status_text.text("配置实体类型...")
                progress_bar.progress(20)
                builder.set_ontology(graph_id)
                
                # 分块
                status_text.text("分割文本...")
                progress_bar.progress(30)
                chunks = split_text_into_chunks(text_content, max_chunk_size=chunk_size)
                st.info(f"文本已分为 {len(chunks)} 个块")
                
                # 添加到图谱
                status_text.text("正在发送数据到Zep...")
                progress_bar.progress(40)
                
                def update_progress(msg):
                    status_text.text(msg)
                
                # 分批发送数据
                task_ids = builder.add_text_to_graph(
                    graph_id=graph_id,
                    text_chunks=chunks,
                    batch_size=3,
                    progress_callback=update_progress
                )
                
                # 等待处理完成
                progress_bar.progress(60)
                status_text.text("等待Zep处理数据...")
                
                if task_ids:
                    builder.wait_for_tasks(
                        task_ids, 
                        timeout=600,
                        progress_callback=update_progress
                    )
                
                # 获取图数据
                status_text.text("获取图谱数据...")
                progress_bar.progress(90)
                st.session_state.graph_data = builder.get_graph_data(graph_id)
                st.session_state.graph_id = graph_id
                
                progress_bar.progress(100)
                status_text.text("完成!")
                st.success(f"知识图谱生成成功! Graph ID: {graph_id}")
                
            except Exception as e:
                st.error(f"生成图谱时出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 显示图谱
    if st.session_state.graph_data:
        graph_data = st.session_state.graph_data
        
        # 统计信息
        display_stats(graph_data)
        
        st.divider()
        
        # 图谱可视化
        st.subheader("知识图谱可视化")
        
        if graph_data.nodes:
            with st.spinner("渲染图谱..."):
                html_content = create_pyvis_graph(graph_data)
                components.html(html_content, height=750, scrolling=True)
        else:
            st.warning("图谱中没有节点")
        
        st.divider()
        
        # 实体列表
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_entity_list(graph_data)
        
        with col2:
            st.subheader("关系列表")
            if graph_data.edges:
                for edge in graph_data.edges[:50]:  # 只显示前50条
                    st.markdown(f"- **{edge.fact}**" if edge.fact else f"- {edge.name}")
                if len(graph_data.edges) > 50:
                    st.caption(f"...还有 {len(graph_data.edges) - 50} 条关系")
            else:
                st.info("暂无关系数据")


if __name__ == "__main__":
    main()

