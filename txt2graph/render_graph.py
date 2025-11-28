"""
Zep Graph HTML渲染器
从Zep云服务获取图谱数据并生成交互式HTML可视化
"""

import os
import json
import argparse
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv
from zep_cloud.client import Zep
from pyvis.network import Network

# 加载环境变量
load_dotenv()

# 实体类型对应的颜色
ENTITY_COLORS = {
    "Person": "#ff6b6b",
    "Company": "#4ecdc4", 
    "Organization": "#45b7d1",
    "Location": "#96ceb4",
    "Product": "#ffeead",
    "Event": "#dcc6e0",
    "Media": "#ffb74d",
    "Preference": "#a29bfe",
    "Topic": "#fd79a8",
    "Object": "#636e72",
    "Entity": "#b2bec3",
}

# 默认颜色
DEFAULT_COLOR = "#74b9ff"


def get_graph_data(client: Zep, graph_id: str, max_retries: int = 3) -> tuple[list, list]:
    """
    从Zep获取图谱的所有节点和边
    """
    print(f"正在获取图谱 {graph_id} 的数据...")
    
    nodes = None
    edges = None
    
    # 获取所有节点（带重试）
    for attempt in range(max_retries):
        try:
            nodes = client.graph.node.get_by_graph_id(graph_id=graph_id)
            print(f"  获取到 {len(nodes)} 个节点")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  获取节点失败，重试中... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                raise e
    
    # 获取所有边（带重试）
    for attempt in range(max_retries):
        try:
            edges = client.graph.edge.get_by_graph_id(graph_id=graph_id)
            print(f"  获取到 {len(edges)} 个边")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  获取边失败，重试中... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                raise e
    
    return nodes or [], edges or []


def create_html_graph(nodes: list, edges: list, graph_id: str, output_file: str):
    """
    创建交互式HTML图谱可视化
    """
    print("正在生成HTML可视化...")
    
    # 创建网络图
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
        notebook=False,
    )
    
    # 配置物理引擎和交互
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial, sans-serif",
                "color": "white"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.5)",
                "size": 10
            }
        },
        "edges": {
            "color": {
                "color": "#555555",
                "highlight": "#667eea",
                "hover": "#667eea"
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
                "color": "#aaaaaa",
                "face": "Arial, sans-serif",
                "strokeWidth": 0,
                "background": "rgba(26,26,46,0.8)"
            },
            "width": 1.5,
            "hoverWidth": 2.5
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.5
            },
            "stabilization": {
                "enabled": true,
                "iterations": 300,
                "updateInterval": 25
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            },
            "multiselect": true,
            "zoomView": true
        }
    }
    """)
    
    # 构建节点UUID到节点的映射
    node_map = {}
    for node in nodes:
        node_map[node.uuid_] = node
    
    # 统计节点类型
    type_counts = {}
    
    # 添加节点
    for node in nodes:
        node_labels = node.labels or []
        specific_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
        node_type = specific_labels[0] if specific_labels else (node_labels[0] if node_labels else "Unknown")
        
        color = ENTITY_COLORS.get(node_type, DEFAULT_COLOR)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # 简化的工具提示（避免复杂HTML）
        summary_text = (node.summary or '无摘要')[:200]
        title = f"{node.name}\n类型: {', '.join(node_labels)}\n摘要: {summary_text}"
        
        # 根据节点类型调整大小
        if node_type == "Person":
            size = 25
        elif node_type in ["Company", "Organization"]:
            size = 30
        elif node_type == "Location":
            size = 22
        elif node_type == "Event":
            size = 20
        else:
            size = 18
        
        net.add_node(
            node.uuid_,
            label=node.name[:20] + "..." if len(node.name) > 20 else node.name,
            title=title,
            color=color,
            size=size,
        )
    
    # 添加边
    edge_count = 0
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        
        if source_uuid in node_map and target_uuid in node_map:
            source_node = node_map[source_uuid]
            target_node = node_map[target_uuid]
            
            title = f"{source_node.name} -> {target_node.name}\n关系: {edge.name or '未命名'}\n事实: {edge.fact or '无描述'}"
            
            edge_label = edge.name or ""
            if len(edge_label) > 15:
                edge_label = edge_label[:15] + "..."
            
            net.add_edge(
                source_uuid,
                target_uuid,
                title=title,
                label=edge_label,
            )
            edge_count += 1
    
    print(f"  添加了 {len(nodes)} 个节点和 {edge_count} 条边到可视化")
    
    # 打印类型统计
    print("\n节点类型统计:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    # 保存到临时文件获取原始HTML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name
    
    # 读取生成的HTML
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    os.unlink(tmp_path)
    
    # 构建自定义CSS和Header HTML
    custom_css = """
        <style>
            #graph-header {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                color: white;
                font-family: Arial, sans-serif;
            }
            #graph-header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 600;
            }
            #graph-header .stats {
                font-size: 0.9rem;
                opacity: 0.9;
                margin-top: 5px;
            }
            #graph-legend {
                position: fixed;
                top: 80px;
                right: 20px;
                z-index: 1000;
                background: rgba(26, 26, 46, 0.95);
                border: 1px solid #333;
                border-radius: 8px;
                padding: 15px;
                max-width: 200px;
                color: white;
                font-family: Arial, sans-serif;
            }
            #graph-legend h3 {
                margin: 0 0 10px 0;
                font-size: 0.9rem;
                border-bottom: 1px solid #444;
                padding-bottom: 5px;
            }
            #graph-legend .item {
                display: flex;
                align-items: center;
                margin: 5px 0;
                font-size: 0.8rem;
            }
            #graph-legend .color-box {
                width: 12px;
                height: 12px;
                border-radius: 3px;
                margin-right: 8px;
                flex-shrink: 0;
            }
            #mynetwork {
                margin-top: 70px !important;
            }
            body {
                background: #1a1a2e !important;
            }
        </style>
    """
    
    # 构建图例HTML
    legend_items = ""
    for entity_type, color in ENTITY_COLORS.items():
        if entity_type in type_counts:
            legend_items += f'''
            <div class="item">
                <div class="color-box" style="background: {color};"></div>
                <span>{entity_type} ({type_counts.get(entity_type, 0)})</span>
            </div>'''
    
    custom_html = f'''
        <div id="graph-header">
            <h1>Knowledge Graph Visualization</h1>
            <div class="stats">
                Graph ID: {graph_id} | 节点: {len(nodes)} | 边: {edge_count} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        <div id="graph-legend">
            <h3>图例</h3>
            {legend_items}
        </div>
    '''
    
    # 在</head>前插入自定义CSS
    html_content = html_content.replace('</head>', custom_css + '</head>')
    
    # 在<body>后插入自定义HTML
    html_content = html_content.replace('<body>', '<body>' + custom_html)
    
    # 修改标题
    html_content = html_content.replace('<title>Network</title>', f'<title>Zep Graph: {graph_id}</title>')
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML文件已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Zep Graph HTML渲染器')
    parser.add_argument(
        '--graph-id', '-g',
        default='graph_6e3697873495400d',
        help='Zep图谱ID'
    )
    parser.add_argument(
        '--output', '-o',
        default='graph_visualization.html',
        help='输出HTML文件路径'
    )
    parser.add_argument(
        '--api-key', '-k',
        default=None,
        help='Zep API Key (默认从环境变量ZEP_API_KEY读取)'
    )
    
    args = parser.parse_args()
    
    # 获取API Key
    api_key = args.api_key or os.environ.get('ZEP_API_KEY')
    if not api_key:
        print("错误: 请设置ZEP_API_KEY环境变量或通过--api-key参数提供")
        return 1
    
    # 创建客户端
    client = Zep(api_key=api_key)
    
    try:
        # 获取图数据
        nodes, edges = get_graph_data(client, args.graph_id)
        
        if not nodes:
            print("警告: 图谱中没有节点")
            return 1
        
        # 生成HTML
        create_html_graph(nodes, edges, args.graph_id, args.output)
        
        print(f"\n完成! 请在浏览器中打开 {args.output} 查看图谱")
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
