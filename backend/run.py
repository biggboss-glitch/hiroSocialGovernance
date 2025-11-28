"""
MiroFish Backend 启动入口
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """主函数"""
    # 验证配置
    errors = Config.validate()
    if errors:
        print("配置错误:")
        for err in errors:
            print(f"  - {err}")
        print("\n请检查 .env 文件中的配置")
        sys.exit(1)
    
    # 创建应用
    app = create_app()
    
    # 获取运行配置
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    print(f"""
╔══════════════════════════════════════════════════╗
║           MiroFish Backend Server                ║
╠══════════════════════════════════════════════════╣
║  Running on: http://{host}:{port}
║  Debug mode: {debug}
║  
║  API Endpoints:
║    POST /api/graph/ontology/generate  - 生成本体
║    POST /api/graph/build              - 构建图谱
║    GET  /api/graph/task/<task_id>     - 查询任务
║    GET  /api/graph/tasks              - 列出任务
║    GET  /api/graph/data/<graph_id>    - 获取图数据
║    DELETE /api/graph/delete/<graph_id>- 删除图谱
╚══════════════════════════════════════════════════╝
    """)
    
    # 启动服务
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

