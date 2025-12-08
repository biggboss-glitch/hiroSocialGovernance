"""
MiroFish Backend 启动入口
"""

import os
import sys
import warnings

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 这个警告是无害的，只是在进程被终止时资源没有被正确清理
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*multiprocessing.*")

# 额外：通过环境变量告诉 Python 不要跟踪 multiprocessing 资源
# 这可以从根本上避免警告
import os
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')

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
    
    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
    # WERKZEUG_RUN_MAIN=true 表示当前是 reloader 启动的子进程
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    if not debug or is_reloader_process:
        print(f"""
╔══════════════════════════════════════════════════╗
║           MiroFish Backend Server                ║
╠══════════════════════════════════════════════════╣
║  Running on: http://{host}:{port}                ║
║  Debug mode: {debug}                             ║
║                                                  ║
║  API Endpoints:                                  ║
║    POST /api/graph/ontology/generate  - 生成本体  ║
║    POST /api/graph/build              - 构建图谱  ║
║    GET  /api/graph/task/<task_id>     - 查询任务  ║
║    GET  /api/graph/tasks              - 列出任务  ║
║    GET  /api/graph/data/<graph_id>    - 获取图数据║
║    DELETE /api/graph/delete/<graph_id>- 删除图谱  ║
╚══════════════════════════════════════════════════╝
        """)
    
    # 启动服务
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

